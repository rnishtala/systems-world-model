"""Empirical trajectory environment calibrated only from real training incidents.
This is a training surrogate, not a claim of live-service evaluation.
"""
import json
from pathlib import Path
import numpy as np
ACTIONS=('wait','inspect_cpu','inspect_pool','inspect_trace','healthy','payment_cpu','ledger_pool','ledger_delay','escalate')
GROUPS={1:(3,4),2:(5,6),3:(7,8,9)}
DIAGNOSES=('healthy','payment:cpu_contention','ledger:pool_contention','ledger:downstream_delay')
GAIN=np.array([10,1,1,3,3,10,4,10,10,10],np.float32)
def normalize(vector):return np.clip(np.asarray(vector,np.float32)*GAIN,0,2).tolist()

class Incident:
    def __init__(self,frames,fault,penalty=2):
        self.frames=frames;self.fault=fault;self.penalty=penalty;self.t=0;self.checks=0;self.done=False;self.choice=None
        self.seen=np.zeros(3,np.float32);self.values=np.zeros(7,np.float32)
    def observe(self):
        f=normalize(self.frames[min(self.t,len(self.frames)-1)]['vector'])
        return np.array([*f[:3],*self.values,*self.seen,self.t/7],np.float32)
    def step(self,action):
        if self.done:raise RuntimeError('episode ended')
        if not 0<=action<9:raise ValueError(action)
        self.t+=1;reward=-.05
        if action in GROUPS:
            self.checks+=1;reward-=.15;self.seen[action-1]=1
            f=normalize(self.frames[min(self.t,len(self.frames)-1)]['vector'])
            for i in GROUPS[action]:self.values[i-3]=f[i]
        elif 4<=action<=7:
            self.choice=action-4;reward+=3 if self.choice==self.fault else -3;self.done=True
        elif action==8:self.choice=-1;reward-=self.penalty;self.done=True
        if self.t>=7 and not self.done:self.choice=-1;reward-=self.penalty;self.done=True
        return self.observe(),reward,self.done

class EmpiricalEnv:
    def __init__(self,path,seed=0):
        import elements
        self.rows=[json.loads(l) for l in Path(path).read_text().splitlines()];self.rng=np.random.default_rng(seed);self.sim=None
        if not self.rows or any(len(r['frames'])<4 for r in self.rows):raise ValueError('Incomplete calibration')
        if any(f['successes']!=f['requests'] for r in self.rows for f in r['frames']):raise ValueError('Unexpected request failures in calibration')
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,9))
    def step(self,action):
        if bool(action['reset']) or self.sim is None or self.sim.done:
            row=self.rows[int(self.rng.integers(len(self.rows)))];self.sim=Incident(row['frames'],row['fault'])
            return dict(vector=self.sim.observe(),reward=np.float32(0),is_first=True,is_last=False,is_terminal=False)
        v,r,d=self.sim.step(int(action['action']));return dict(vector=v,reward=np.float32(r),is_first=False,is_last=d,is_terminal=d)
    def close(self):pass

class Coverage:
    def __init__(self,policy,seed):self.policy=policy;self.rng=np.random.default_rng(seed);self.schedules={};self.stats={'guided':0,'policy':0}
    def __call__(self,carry,obs,mode='train'):
        carry,acts,outs=self.policy(carry,obs,mode=mode)
        if mode!='train':return carry,acts,outs
        actions=acts['action'].copy()
        for i,first in enumerate(obs['is_first']):
            if first:
                guided=self.rng.random()<.5;self.stats['guided' if guided else 'policy']+=1
                self.schedules[i]=list(self.rng.permutation([1,2,3]))+[int(self.rng.integers(4,8))] if guided else []
            if self.schedules.get(i) and not obs['is_last'][i]:actions[i]=self.schedules[i].pop(0)
        import jax
        carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.int32(a),ref.sharding) for a,ref in zip(actions,carry[-1]['action'])]})
        return carry,{**acts,'action':actions},outs
