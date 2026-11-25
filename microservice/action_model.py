"""Measured macro-transition replay for action-conditioned upstream DreamerV3.

Four telemetry frames -> one post-action outcome frame. No imagined outcomes are
used as training examples, and held-out/diagnostic rows are rejected.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.action_outcomes import ACTIONS,TARGETS
from microservice.temporal_calibration import FEATURES

VERSION='repair-macro-v1'
WIDTH=len(FEATURES)+len(TARGETS)+2

def validate(rows):
    if not rows:raise ValueError('Empty training partition')
    for r in rows:
        if r.get('split')!='train' or not r.get('restored'):raise ValueError('Only restored training trials allowed')
        if r.get('action') not in ACTIONS:raise ValueError('Unknown action')
        h=np.asarray(r['observations'],float);y=np.asarray(r['outcome'],float)
        if h.shape!=(4,len(FEATURES)) or y.shape!=(4,) or not np.isfinite(h).all() or not np.isfinite(y).all() or (h<0).any() or (y<0).any() or y[0]>1:raise ValueError('Invalid measured trajectory')
    return rows

def export_training(source,out):
    source=Path(source);out=Path(out)
    protocol=json.loads((source/'protocol.json').read_text())
    summary=json.loads((source/'summary.json').read_text())['action_outcomes']
    raw=(source/'examples.json').read_bytes();rows=json.loads(raw)
    if not summary['all_restored'] or len(rows)!=len(protocol['cases']):raise ValueError('Incomplete source')
    for r,c in zip(rows,protocol['cases']):
        if any(r[k]!=c[k] for k in ('split','group','action')):raise ValueError('Source schedule mismatch')
    train=validate([{k:r[k] for k in ('split','group','action','observations','outcome','restored')} for r in rows if r['split']=='train'])
    if len(train)!=summary['training_trials']:raise ValueError('Training count mismatch')
    out.mkdir(parents=True,exist_ok=False)
    data=''.join(json.dumps(r)+'\n' for r in train).encode();(out/'train.jsonl').write_bytes(data)
    meta=dict(version=VERSION,source_run=source.parent.name,source_examples_sha256=hashlib.sha256(raw).hexdigest(),training_sha256=hashlib.sha256(data).hexdigest(),training_trials=len(train),training_profiles=len({r['group'] for r in train}),excluded_trials=len(rows)-len(train),features=FEATURES,targets=TARGETS,actions=ACTIONS,scope='Training smoke test; macro-transition model; no held-out performance claim')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));return meta

def frame(row,t):
    if not 0<=t<=4:raise ValueError('Invalid frame')
    v=np.zeros(WIDTH,np.float32)
    if t<4:v[:len(FEATURES)]=np.log1p(row['observations'][t]);v[-2]=1
    else:v[len(FEATURES):len(FEATURES)+4]=np.log1p(row['outcome']);v[-1]=1
    return v

def next_action(row,t):
    # The repair is applied only after all four observed prefix frames.
    return np.eye(4,dtype=np.float32)[ACTIONS.index(row['action']) if t==3 else 0]

class ActionEnv:
    def __init__(self,path,seed=0):
        import elements
        self.rows=validate([json.loads(l) for l in Path(path).read_text().splitlines()]);self.rng=np.random.default_rng(seed);self.done=True;self.t=0
        self.obs_space=dict(vector=elements.Space(np.float32,(WIDTH,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        # Replay scheduling metadata is explicitly excluded from model spaces and policy inputs.
        self.obs_space['scheduled_action']=elements.Space(np.float32,(4,),0,1)
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.float32,(4,),0,1))
    def step(self,action):
        first=bool(action['reset']) or self.done
        if first:self.row=self.rows[int(self.rng.integers(len(self.rows)))];self.t=0;self.done=False
        else:
            if np.asarray(action['action']).shape!=(4,) or not np.allclose(action['action'],next_action(self.row,self.t),atol=1e-6):raise ValueError('Unmeasured action; counterfactual replay forbidden')
            self.t+=1;self.done=self.t==4
        return dict(vector=frame(self.row,self.t),reward=np.float32(0),is_first=first,is_last=self.done,is_terminal=self.done,**{'scheduled_action':next_action(self.row,self.t)})
    def close(self):pass

class MeasuredActions:
    def __init__(self,policy):self.policy=policy;self.steps=0
    def __call__(self,carry,obs,mode='train'):
        import jax
        clean={k:v for k,v in obs.items() if k!='scheduled_action'}
        carry,acts,outs=self.policy(carry,clean,mode=mode)
        actions=2*np.asarray(obs['scheduled_action'],np.float32)-1
        self.steps+=len(actions)
        carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(a,ref.sharding) for a,ref in zip(actions,carry[-1]['action'])]})
        return carry,{**acts,'action':actions},outs


def model_spaces(env):
    return ({k:v for k,v in env.obs_space.items() if k!='scheduled_action'},
            {k:v for k,v in env.act_space.items() if k!='reset'})

def make_agent(config,path):
    from dreamerv3.agent import Agent
    import elements
    env=ActionEnv(path);obs,acts=model_spaces(env);env.close()
    return Agent(obs,acts,elements.Config(**config.agent,logdir=config.logdir,seed=config.seed,jax=config.jax,batch_size=config.batch_size,batch_length=config.batch_length,replay_context=config.replay_context,report_length=config.report_length,replica=config.replica,replicas=config.replicas))
