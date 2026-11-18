"""Trace-only sequence view: identical observations for Dreamer and baselines."""
import json
from pathlib import Path
import numpy as np
from microservice.environment import Incident, EmpiricalEnv, normalize
from microservice.mechanisms import FAULT_MASK


def vector(row, t):
    f = normalize(row['frames'][min(t,3)]['vector'])
    # Fixed six observable features; direct resource counters never enter this view.
    return np.array([f[0],f[1],f[2],f[7],f[8],f[9],0,0,0,0,
                     *([float(t>=3)]*3),min(t,7)/7],np.float32)


class TemporalIncident(Incident):
    def observe(self):
        return vector({'frames':self.frames},self.t)


class TemporalEnv(EmpiricalEnv):
    def __init__(self,path,seed=0,compositional=True):
        super().__init__(path,seed,compositional)
        import elements
        self.obs_space['mechanism_target']=elements.Space(np.float32,(3,),0,1)

    def step(self,action):
        new = bool(action['reset']) or self.sim is None or self.sim.done
        obs = super().step(action)
        if new:
            self.sim=TemporalIncident(self.sim.frames,self.sim.fault,compositional=True)
            obs['vector']=self.sim.observe()
        mask=FAULT_MASK[self.sim.fault]
        obs['mechanism_target']=np.array([(mask>>i)&1 for i in range(3)],np.float32)
        return obs


class TemporalCoverage:
    def __init__(self,policy,seed,compositional=True):
        self.policy=policy;self.rng=np.random.default_rng(seed);self.schedules={}
        self.stats={'guided':0,'policy':0}
    def __call__(self,carry,obs,mode='train'):
        carry,acts,outs=self.policy(carry,obs,mode=mode)
        if mode!='train':return carry,acts,outs
        actions=acts['action'].copy()
        for i,first in enumerate(obs['is_first']):
            if first:
                guided=self.rng.random()<.5;self.stats['guided' if guided else 'policy']+=1
                self.schedules[i]=[0,0,0,int(self.rng.integers(4,12))] if guided else []
            if self.schedules.get(i) and not obs['is_last'][i]:actions[i]=self.schedules[i].pop(0)
        import jax
        carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.int32(a),ref.sharding) for a,ref in zip(actions,carry[-1]['action'])]})
        return carry,{**acts,'action':actions},outs


def extract(model,rows):
    import jax
    from microservice.order_features import force_action
    history=[];latent=[];aux=[]
    for i,row in enumerate(rows):
        carry=model.policy.init_policy(1);sequence=[]
        for t in range(4):
            v=vector(row,t);sequence.append(v)
            obs=dict(vector=v[None],reward=np.array([0. if t==0 else -.05],np.float32),
                     is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),
                     mechanism_target=np.zeros((1,3),np.float32))
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                carry=force_action(carry,0,jax.device_put)
        with jax.transfer_guard('allow'):
            dyn=model.agent._stack(carry)[1]
            state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            probs=np.asarray(model.mechanisms(model.agent.params,dyn,jax.random.PRNGKey(8604+i)))
        history.append(np.asarray(sequence).reshape(-1));latent.append(state)
        aux.append(int(np.sum((probs>.5)*np.array([1,2,4]))))
        if i%20==0:print('TEMPORAL_EXTRACT',i,flush=True)
    history=np.asarray(history)
    return dict(history=history,latest=history[:,-14:],latent=np.asarray(latent)),np.asarray(aux)
