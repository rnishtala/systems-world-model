"""Run a saved policy on simulated incidents; no external actions or LLM calls."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/work/upstream'))
import elements
from dreamerv3 import main as upstream
from experiment.environment import DreamerEnv,QueueTriage
from experiment.particles import ParticleActor
from experiment.greedy import GreedyActor

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--fault',type=int,choices=[0,1,2],default=2);p.add_argument('--all-faults',action='store_true')
p.add_argument('--seed',type=int,default=98765);p.add_argument('--controller',choices=['sampled','greedy','particles'],default='sampled')
p.add_argument('--out');a=p.parse_args()
if a.out and Path(a.out).exists():p.error('Choose a new output path to preserve existing evidence')
c=elements.Config.load(elements.Path(a.config))
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(DreamerEnv(a.seed),config)
agent=upstream.make_agent(c)
cp=elements.Checkpoint(elements.Path(a.checkpoint));cp.agent=agent;cp.load(keys=['agent'])
policy=ParticleActor(agent) if a.controller=='particles' else GreedyActor(agent) if a.controller=='greedy' else agent
cases=[]
for fault in range(3) if a.all_faults else [a.fault]:
    state=policy.init_policy(1);env=QueueTriage(a.seed+fault);obs=env.reset(fault)
    first=True;reward=0.;trace=[]
    while not env.done:
        inp=dict(vector=obs[None],reward=np.array([reward],np.float32),is_first=np.array([first]),
                 is_last=np.array([False]),is_terminal=np.array([False]))
        state,actions,_=policy.policy(state,inp,mode='eval');act=int(np.asarray(actions['action'])[0])
        trace.append({'observations':obs.tolist(),'action':env.ACTIONS[act]})
        obs,reward,_=env.step(act);first=False
    cases.append({'trace':trace,'diagnosis':env.choice,'ground_truth':fault,
                  'correct':env.choice==fault,'checks':env.checks})
result={'scope':'Synthetic incidents only; fault labels are used solely for setup and scoring.',
        'controller':a.controller,'cases':cases}
if a.out:Path(a.out).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
