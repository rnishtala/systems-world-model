"""Measured trajectory replay with continuous, correctly aligned workload actions."""
import json
from pathlib import Path
import numpy as np
from microservice.regime import vector as trace_vector
from microservice.mechanisms import FAULT_MASK

def next_load(row,t):
    return float(row['frames'][t+1]['requests'])/12 if t<3 else 0.

def vector(row,t):
    x=trace_vector(row,min(t,3)).copy()
    # The next workload is a declared experiment schedule, never future telemetry.
    x[6]=next_load(row,t) if t<4 else 0.
    return x

class WorkloadEnv:
    def __init__(self,path,seed=0,compositional=True):
        import elements
        self.rows=[json.loads(l) for l in Path(path).read_text().splitlines()]
        if not self.rows or any(len(r['frames'])!=4 or any(f['requests']!=f['successes'] for f in r['frames']) for r in self.rows):raise ValueError('Invalid measured trajectories')
        self.rng=np.random.default_rng(seed);self.row=None;self.t=0;self.done=True
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),mechanism_target=elements.Space(np.float32,(3,),0,1),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.float32,(),0,1))
    def step(self,action):
        first=bool(action['reset']) or self.done
        if first:
            self.row=self.rows[int(self.rng.integers(len(self.rows)))];self.t=0;self.done=False
        else:
            expected=next_load(self.row,self.t)
            if not np.isclose(float(action['action']),expected,atol=1e-6):raise ValueError('Action differs from measured workload; counterfactual replay forbidden')
            self.t+=1;self.done=self.t==4
        mask=FAULT_MASK[self.row['fault']]
        return dict(vector=vector(self.row,self.t),mechanism_target=np.array([(mask>>i)&1 for i in range(3)],np.float32),reward=np.float32(0),is_first=first,is_last=self.done,is_terminal=self.done)
    def close(self):pass

class WorkloadCoverage:
    def __init__(self,policy,seed,compositional=True):self.policy=policy;self.stats={'scheduled':0}
    def __call__(self,carry,obs,mode='train'):
        carry,acts,outs=self.policy(carry,obs,mode=mode)
        if mode!='train':return carry,acts,outs
        import jax
        actions=2*np.asarray(obs['vector'][:,6],np.float32)-1
        self.stats['scheduled']+=len(actions)
        carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.float32(a),ref.sharding) for a,ref in zip(actions,carry[-1]['action'])]})
        return carry,{**acts,'action':actions},outs

def extract(model,rows):
    import jax
    from microservice.load_response import response
    history=[];latent=[];aux=[];forecasts=[];actuals=[];persistence=[]
    for i,row in enumerate(rows):
        carry=model.policy.init_policy(1);sequence=[]
        for t in range(4):
            v=vector(row,t);sequence.append(v)
            with jax.transfer_guard('allow'):
                if t:assert all(np.isclose(float(np.asarray(a)),2*row['frames'][t]['requests']/12-1) for a in carry[-1]['action'])
            obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),mechanism_target=np.zeros((1,3),np.float32))
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                action=2*next_load(row,t)-1
                carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.float32(action),ref.sharding) for ref in carry[-1]['action']]})
                dyn=model.agent._stack(carry)[1]
                if t<3:
                    prediction=np.asarray(model.forecast(model.agent.params,dyn,np.float32(action),jax.random.PRNGKey(8804+i*4+t)))
                    forecasts.append(prediction[[0,3,4,5]]);actuals.append(vector(row,t+1)[[0,3,4,5]]);persistence.append(v[[0,3,4,5]])
        with jax.transfer_guard('allow'):
            state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            probs=np.asarray(model.mechanisms(model.agent.params,dyn,jax.random.PRNGKey(8804+i)))
        history.append(np.asarray(sequence).reshape(-1));latent.append(state);aux.append(int(np.sum((probs>.5)*np.array([1,2,4]))))
        if i%20==0:print('WORKLOAD_EXTRACT',i,flush=True)
    history=np.asarray(history)
    print('WORKLOAD_ACTIONS_VERIFIED',len(rows)*3,flush=True)
    forecast=dict(predicted=np.asarray(forecasts),actual=np.asarray(actuals),persistence=np.asarray(persistence))
    return dict(history=history,latest=history[:,-14:],response=np.array([response(r) for r in rows]),latent=np.asarray(latent)),np.asarray(aux),forecast

class WorkloadActor:
    """Particle posterior inference; actions are supplied by the measured schedule."""
    def __init__(self,agent):self.agent=agent
    def init_policy(self,n):
        if n!=1:raise ValueError('One external environment required')
        return self.agent.init_policy(16)
    def policy(self,carry,obs,mode='eval'):
        if mode!='eval':raise ValueError('Evaluation only')
        carry,acts,outs=self.agent.policy(carry,{k:np.repeat(v,16,axis=0) for k,v in obs.items()},mode=mode)
        return carry,{'action':np.array([np.asarray(acts['action']).mean()],np.float32)},outs
