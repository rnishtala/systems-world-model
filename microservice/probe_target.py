"""Measured two-dimensional probe action replay and posterior extraction."""
import json
from pathlib import Path
import numpy as np
from microservice.mechanisms import FAULT_MASK

def frames(row):
    result=row['base']+[f for target in row['order'] for f in row['extra'][target]]
    if len(result)!=8 or any(f['requests']!=f['successes'] or f['target'] not in ('checkout','ledger') for f in result):raise ValueError('Invalid measured probe trajectory')
    return result

def next_action(row,t):
    if t>=7:return np.zeros(2,np.float32)
    f=frames(row)[t+1]
    return np.array([float(f['target']=='ledger'),f['requests']/12],np.float32)

def vector(row,t):
    f=frames(row)[min(t,7)];m=f['metrics'];upstream=f['target']=='checkout'
    latency=[m['ledger']['latency_p95'],m['payment']['self_p95'] if upstream else 0.,
             m['checkout']['latency_p95'] if upstream else 0.,m['payment']['downstream_p95'] if upstream else 0.]
    if not np.isfinite(latency).all() or min(latency)<0:raise ValueError('Invalid latency')
    return np.array([*np.log1p(np.array(latency)/.1),float(not upstream),f['requests']/12,
        *next_action(row,t),float(upstream),min(t,7)/7,*([float(t>=7)]*3),m['ledger']['error_rate']],np.float32)

class ProbeEnv:
    def __init__(self,path,seed=0,compositional=True):
        import elements
        self.rows=[json.loads(l) for l in Path(path).read_text().splitlines()]
        if not self.rows:raise ValueError('Empty trajectories')
        for row in self.rows:frames(row)
        self.rng=np.random.default_rng(seed);self.done=True;self.t=0
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),mechanism_target=elements.Space(np.float32,(3,),0,1),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.float32,(2,),0,1))
    def step(self,action):
        first=bool(action['reset']) or self.done
        if first:self.row=self.rows[int(self.rng.integers(len(self.rows)))];self.t=0;self.done=False
        else:
            a=np.asarray(action['action'])
            if a.shape!=(2,) or not np.allclose(a,next_action(self.row,self.t),atol=1e-6):raise ValueError('Unmeasured probe action; counterfactual replay forbidden')
            self.t+=1;self.done=self.t==8
        mask=FAULT_MASK[self.row['fault']]
        return dict(vector=vector(self.row,self.t),mechanism_target=np.array([(mask>>i)&1 for i in range(3)],np.float32),reward=np.float32(0),is_first=first,is_last=self.done,is_terminal=self.done)
    def close(self):pass

class ProbeCoverage:
    def __init__(self,policy,seed,compositional=True):self.policy=policy;self.stats={'scheduled':0}
    def __call__(self,carry,obs,mode='train'):
        carry,acts,outs=self.policy(carry,obs,mode=mode)
        if mode!='train':return carry,acts,outs
        import jax
        actions=2*np.asarray(obs['vector'][:,6:8],np.float32)-1
        self.stats['scheduled']+=len(actions)
        carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(a,ref.sharding) for a,ref in zip(actions,carry[-1]['action'])]})
        return carry,{**acts,'action':actions},outs

class ProbeActor:
    def __init__(self,agent):self.agent=agent
    def init_policy(self,n):
        if n!=1:raise ValueError('One external environment required')
        return self.agent.init_policy(16)
    def policy(self,carry,obs,mode='eval'):
        if mode!='eval':raise ValueError('Evaluation only')
        carry,acts,outs=self.agent.policy(carry,{k:np.repeat(v,16,axis=0) for k,v in obs.items()},mode=mode)
        return carry,{'action':np.asarray(acts['action']).mean(0,keepdims=True)},outs

def extract(model,rows):
    import jax
    from scripts.evaluate_ledger_probe import features as raw_features
    history=[];latents=[];aux=[];predicted=[];actual=[];persistence=[];switch=[]
    for i,row in enumerate(rows):
        carry=model.policy.init_policy(1);sequence=[];fs=frames(row)
        for t in range(8):
            v=vector(row,t);sequence.append(v)
            with jax.transfer_guard('allow'):
                if t:
                    expected=2*np.array([float(fs[t]['target']=='ledger'),fs[t]['requests']/12],np.float32)-1
                    for a in carry[-1]['action']:np.testing.assert_allclose(np.asarray(a),expected,atol=1e-6)
            obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),mechanism_target=np.zeros((1,3),np.float32))
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                action=2*next_action(row,t)-1
                carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(action,ref.sharding) for ref in carry[-1]['action']]})
                dyn=model.agent._stack(carry)[1]
                if t<7:
                    f=np.asarray(model.forecast(model.agent.params,dyn,action,jax.random.PRNGKey(9204+i*8+t)))
                    predicted.append(float(f[0]));actual.append(float(vector(row,t+1)[0]));persistence.append(float(v[0]));switch.append(fs[t]['target']!=fs[t+1]['target'])
        with jax.transfer_guard('allow'):
            state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            probs=np.asarray(model.mechanisms(model.agent.params,dyn,jax.random.PRNGKey(9204+i)))
        history.append(np.array(sequence).reshape(-1));latents.append(state);aux.append(int(np.sum((probs>.5)*np.array([1,2,4]))))
        if i%8==0:print('PROBE_EXTRACT',i,flush=True)
    history=np.array(history);response=raw_features(rows)['ledger']
    xs=dict(latent=np.array(latents),history=history,response=response,response_history=np.concatenate([response,history],1))
    forecast=dict(predicted=np.array(predicted),actual=np.array(actual),persistence=np.array(persistence),switched=np.array(switch))
    assert all(np.isfinite(x).all() for x in [*xs.values(),*forecast.values()])
    print('PROBE_ACTION_CARRY_VERIFIED',len(rows)*7,flush=True)
    return xs,np.array(aux),forecast
