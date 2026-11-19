"""Measured two-dimensional probe action replay and posterior extraction."""
import json
from pathlib import Path
import numpy as np
from microservice.readout import FAULT_MASK

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
