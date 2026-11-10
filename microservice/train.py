"""Train upstream Dreamer against measured training trajectories only."""
import argparse,hashlib,json,os,time
from pathlib import Path
from experiment.run import configuration
from microservice.environment import EmpiricalEnv,Coverage
import elements,embodied
from dreamerv3 import main as upstream
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=100000);p.add_argument('--compositional',action='store_true');a=p.parse_args()
out=Path(a.out);log=out/f'train-{a.seed}'
if log.exists():raise SystemExit('Choose fresh output')
log.mkdir(parents=True);c=configuration(a.seed,log,a.steps)
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(EmpiricalEnv(a.data,config.seed*1000+index,a.compositional),config)
captured={}
def build():
 agent=upstream.make_agent(c);coverage=Coverage(agent.policy,a.seed+9000,a.compositional);agent.policy=coverage;captured.update(agent=agent,coverage=coverage);return agent
args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
c.save(elements.Path(c.logdir)/'config.yaml');start=time.time()
embodied.run.train(build,lambda:upstream.make_replay(c,'replay'),lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),lambda:upstream.make_logger(c),args)
cp=elements.Checkpoint(elements.Path(out/f'final-{a.seed}'));cp.agent=captured['agent'];cp.save()
result=dict(action_contract='mechanisms-v1' if a.compositional else 'single-v1',observation_version='physical-v2',seed=a.seed,steps=a.steps,seconds=time.time()-start,calibration_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),coverage=captured['coverage'].stats,scope='Empirical surrogate training; no live test episodes used.')
(out/f'training-{a.seed}.json').write_text(json.dumps(result,indent=2));print('TRAINING_COMPLETE',json.dumps(result),flush=True)
