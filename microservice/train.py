"""Train upstream Dreamer against measured training trajectories only."""
import argparse,hashlib,json,os,time
from pathlib import Path
from experiment.run import configuration
from microservice.environment import EmpiricalEnv,Coverage
import elements,embodied
from dreamerv3 import main as upstream
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=100000);p.add_argument('--compositional',action='store_true');p.add_argument('--mechanism-supervision',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--observation-view',choices=['physical-v2','trace-temporal-v1','trace-log-v1','trace-workload-v1'],default='physical-v2');a=p.parse_args()
if a.mechanism_supervision and not a.compositional:p.error('--mechanism-supervision requires --compositional')
from microservice.supervision import SupervisedEnv,make_agent
env_class=SupervisedEnv if a.mechanism_supervision else EmpiricalEnv
agent_factory=make_agent if a.mechanism_supervision else upstream.make_agent
if a.observation_view=='trace-temporal-v1':
 if not a.mechanism_supervision:p.error('Temporal view requires mechanism supervision')
 from microservice.temporal import TemporalEnv,TemporalCoverage
 env_class=TemporalEnv;Coverage=TemporalCoverage
if a.observation_view=='trace-log-v1':
 if not a.mechanism_supervision:p.error('Log view requires mechanism supervision')
 from microservice.regime import LogEnv
 from microservice.temporal import TemporalCoverage
 env_class=LogEnv;Coverage=TemporalCoverage
if a.observation_view=='trace-workload-v1':
 if not a.mechanism_supervision:p.error('Workload view requires mechanism supervision')
 from microservice.workload import WorkloadEnv,WorkloadCoverage
 env_class=WorkloadEnv;Coverage=WorkloadCoverage
out=Path(a.out);log=out/f'train-{a.seed}'
if log.exists() and not a.resume:raise SystemExit('Choose fresh output or explicitly --resume')
if a.resume and not (log/'config.yaml').exists():raise SystemExit('Resume requires existing configuration')
log.mkdir(parents=True,exist_ok=a.resume);c=configuration(a.seed,log,a.steps)
identity=dict(seed=a.seed,steps=a.steps,compositional=a.compositional,mechanism_supervision=a.mechanism_supervision,calibration_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest())
if a.observation_view!='physical-v2':identity['observation_view']=a.observation_view
identity_path=out/f'identity-{a.seed}.json'
if a.resume:
 if not identity_path.exists() or json.loads(identity_path.read_text())!=identity:raise SystemExit('Resume identity mismatch')
 saved=elements.Config.load(elements.Path(log/'config.yaml'))
 if saved!=c:raise SystemExit('Resume configuration mismatch')
else:identity_path.write_text(json.dumps(identity,indent=2))
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(env_class(a.data,config.seed*1000+index,a.compositional),config)
captured={}
def build():
 agent=agent_factory(c);coverage=Coverage(agent.policy,a.seed+9000,a.compositional);agent.policy=coverage;captured.update(agent=agent,coverage=coverage);return agent
args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
c.save(elements.Path(c.logdir)/'config.yaml');start=time.time()
embodied.run.train(build,lambda:upstream.make_replay(c,'replay'),lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),lambda:upstream.make_logger(c),args)
cp=elements.Checkpoint(elements.Path(out/f'final-{a.seed}'));cp.agent=captured['agent'];cp.save()
result=dict(resumed=a.resume,timing_scope="Current process only; interrupted training time excluded" if a.resume else "Complete training process",mechanism_supervision=a.mechanism_supervision,auxiliary_loss_weight=1.0 if a.mechanism_supervision else 0.0,action_contract='workload-continuous-v1' if a.observation_view=='trace-workload-v1' else ('mechanisms-v1' if a.compositional else 'single-v1'),observation_version=a.observation_view,seed=a.seed,steps=a.steps,seconds=time.time()-start,calibration_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),coverage=captured['coverage'].stats,scope='Empirical surrogate training; no live test episodes used.')
(out/f'training-{a.seed}.json').write_text(json.dumps(result,indent=2));print('TRAINING_COMPLETE',json.dumps(result),flush=True)
