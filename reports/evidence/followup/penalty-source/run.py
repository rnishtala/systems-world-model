"""Run upstream DreamerV3 with an injected environment factory, then blind evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, os.environ.get('DREAMER_SOURCE', '/work/upstream'))
from experiment.environment import DreamerEnv, QueueTriage
import elements
import embodied
from dreamerv3 import main as upstream
import ruamel.yaml as yaml


def configuration(seed, logdir, steps):
    configs = yaml.YAML(typ='safe').load(Path(upstream.__file__).with_name('configs.yaml').read_text())
    c = elements.Config(configs['defaults']).update(configs['size1m'])
    return c.update(dict(seed=seed, logdir=str(logdir), task='triage_queue', batch_size=8,
        batch_length=16, report_length=16, **{
        'jax.platform':'cpu', 'jax.compute_dtype':'float32', 'jax.prealloc':False,
        'run.steps':steps, 'run.envs':1, 'run.train_ratio':16., 'run.log_every':30,
        'run.report_every':300, 'run.save_every':30, 'run.debug':True,
        'run.usage.nvsmi':False, 'replay.size':10000,
        'logger.outputs':['jsonl'], 'agent.report':False}))


def evaluation(agent, seed, shifted, n=150):
    rows=[]
    for method in ('dreamer', 'checklist', 'adaptive', 'random'):
        sim=QueueTriage(seed, shifted)
        rng=np.random.default_rng(seed+7)
        carry=agent.init_policy(1) if method=='dreamer' else None
        for episode in range(n):
            obs=sim.reset(fault=episode%3)
            first=True
            reward=0.
            total=0.
            while not sim.done:
                if method=='dreamer':
                    inp=dict(vector=obs[None], reward=np.array([reward],np.float32),
                        is_first=np.array([first]), is_last=np.array([False]), is_terminal=np.array([False]))
                    carry, acts, _ = agent.policy(carry, inp, mode='eval')
                    action=int(np.asarray(acts['action'])[0])
                elif method=='checklist':
                    action=1 if not sim.cpu_seen else (2 if not sim.dep_seen else
                        (4 if sim.cpu>.65 else 5 if sim.dep>.55 else 3))
                elif method=='adaptive':
                    action=1 if not sim.cpu_seen else (4 if sim.cpu>.65 else
                        2 if not sim.dep_seen else 5 if sim.dep>.55 else 3)
                else:
                    action=int(rng.integers(7))
                obs,reward,_=sim.step(action)
                total+=reward
                first=False
            rows.append(dict(method=method, episode=episode, fault=sim.fault, diagnosis=sim.choice,
                correct=sim.choice==sim.fault, checks=sim.checks, steps=sim.t, reward=total,
                shifted=shifted, eval_seed=seed))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--out',default='/results')
    p.add_argument('--abstention-penalty',type=float,default=.5)
    a=p.parse_args();out=Path(a.out)
    if (out/f'train-{a.seed}').exists() or (out/f'results-{a.seed}.json').exists():
        p.error('This seed already has output; choose a fresh --out directory')
    out.mkdir(parents=True,exist_ok=True)
    c=configuration(a.seed,out/f'train-{a.seed}',a.steps)
    upstream.make_env=lambda config,index,**kw: upstream.wrap_env(DreamerEnv(config.seed*1000+index, abstention_penalty=a.abstention_penalty),config)
    captured={}
    make=upstream.make_agent
    def build():
        agent=make(c);captured['agent']=agent;return agent
    args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,
        batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,
        consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
    Path(c.logdir).mkdir(parents=True,exist_ok=True);c.save(elements.Path(c.logdir)/'config.yaml')
    started=time.time()
    embodied.run.train(build,lambda:upstream.make_replay(c,'replay'),
        lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),
        lambda:upstream.make_logger(c),args)
    train_seconds=time.time()-started
    cp=elements.Checkpoint(elements.Path(out/f'final-{a.seed}'));cp.agent=captured['agent'];cp.save()
    offset=int(os.environ.get('EVAL_SEED_OFFSET','100000'))
    rows=evaluation(captured['agent'],offset+a.seed,False)
    rows+=evaluation(captured['agent'],offset+100000+a.seed,True)
    result=dict(training_seed=a.seed,steps=a.steps,train_seconds=train_seconds,
        training_abstention_penalty=a.abstention_penalty, evaluation_abstention_penalty=.5,
        upstream_revision='e01491fad6434b2245a3b8ca201dd7faedcc458c',rows=rows)
    (out/f'results-{a.seed}.json').write_text(json.dumps(result,indent=2))
    print('EXPERIMENT_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
