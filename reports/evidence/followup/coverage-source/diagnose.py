"""Replay an actual trained policy on one simulated incident; no external actions."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/work/upstream'))
import elements
from dreamerv3 import main as upstream
from experiment.environment import DreamerEnv,QueueTriage

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--fault',type=int,choices=[0,1,2],default=2);p.add_argument('--seed',type=int,default=98765)
a=p.parse_args()
c=elements.Config.load(elements.Path(a.config))
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(DreamerEnv(a.seed),config)
agent=upstream.make_agent(c)
cp=elements.Checkpoint(elements.Path(a.checkpoint));cp.agent=agent;cp.load(keys=['agent'])
state=agent.init_policy(1);env=QueueTriage(a.seed);obs=env.reset(a.fault)
first=True;reward=0.;trace=[]
while not env.done:
    inp=dict(vector=obs[None],reward=np.array([reward],np.float32),is_first=np.array([first]),
             is_last=np.array([False]),is_terminal=np.array([False]))
    state,actions,_=agent.policy(state,inp,mode='eval');act=int(np.asarray(actions['action'])[0])
    trace.append({'observations':obs.tolist(),'action':env.ACTIONS[act]})
    obs,reward,_=env.step(act);first=False
print(json.dumps({'trace':trace,'diagnosis':env.choice,'ground_truth':a.fault,
                  'correct':env.choice==a.fault,'checks':env.checks},indent=2))
