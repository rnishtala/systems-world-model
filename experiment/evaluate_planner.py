"""Evaluate a separate lookahead controller; preserve native-actor evidence."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/results/work/upstream'))
import elements
from dreamerv3 import main as upstream
from experiment.environment import DreamerEnv
from experiment.run import evaluation
from experiment.planner import WorldModelPlanner
from experiment.greedy import GreedyActor
from experiment.particles import ParticleActor
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=150);p.add_argument('--eval-offset',type=int,default=7000000)
p.add_argument('--controller',choices=['lookahead','greedy','particles'],default='lookahead')
a=p.parse_args()
if Path(a.out).exists():p.error('Choose a fresh output file; existing evidence is preserved')
c=elements.Config.load(elements.Path(a.config))
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(DreamerEnv(0),config)
agent=upstream.make_agent(c);cp=elements.Checkpoint(elements.Path(a.checkpoint));cp.agent=agent;cp.load(keys=['agent'])
planner=(ParticleActor(agent) if a.controller=='particles' else GreedyActor(agent) if a.controller=='greedy' else WorldModelPlanner(agent,c.seed+8000))
method={'particles':'dreamer_particle_actor','greedy':'dreamer_greedy_actor','lookahead':'world_model_planner'}[a.controller]
start=time.time()
rows=evaluation(planner,a.eval_offset+c.seed,False,a.episodes)+evaluation(planner,a.eval_offset+100000+c.seed,True,a.episodes)
for r in rows:
    if r['method']=='dreamer':r['method']=method
result=dict(training_seed=c.seed,evaluation_seconds=time.time()-start,
            method=method,controller_notes={'particles':'Vote over 16 native actor latent-state particles; no extra inspections or expert action rules.','greedy':'Native actor argmax; latent state remains stochastic; no action bounds added.','lookahead':'Frozen world-model lookahead; each check and wait capped at once by harness rules.'}[a.controller],rows=rows)
Path(a.out).write_text(json.dumps(result,indent=2))
import statistics
for shift in (False,True):
    x=[r for r in rows if r['method']==method and r['shifted']==shift]
    print('PLANNER_RESULT',json.dumps(dict(shifted=shift,accuracy=statistics.mean(r['correct'] for r in x),checks=statistics.mean(r['checks'] for r in x),unresolved=statistics.mean(r['diagnosis']==-1 for r in x))),flush=True)
