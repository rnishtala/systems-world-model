"""Immutable per-run process: train, freeze, collect fresh test, evaluate."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from launcher.common import dataset_check,NAMESPACE

child=None

def stop(signum,frame):
    if child is not None and child.poll() is None:child.terminate()
    raise SystemExit(128+signum)

def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--test-seed',type=int,required=True);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE:raise RuntimeError('Wrong namespace')
    import re
    if not re.fullmatch('[a-f0-9]{12}',a.run):raise ValueError('Invalid run ID')
    root=Path('/results/experiments/runs')/a.run;root.mkdir(parents=True,exist_ok=True)
    if (root/'models').exists():raise RuntimeError('Refusing to overwrite an existing training attempt')
    data=dataset_check(a.dataset)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def phase(name,args):
        global child
        (root/'progress.json').write_text(json.dumps(dict(phase=name,utc=time.time())))
        print('EXPERIMENT_PHASE',name,flush=True)
        with (root/'run.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-m',*args],stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        if code:raise RuntimeError(f'{name} exited with status {code}')
    try:
        phase('training',['microservice.train','--data',str(data),'--out',str(root/'models'),'--seed',str(a.seed),'--steps',str(a.steps)])
        phase('fitting readouts',['microservice.validate','fit','--data',str(data),'--models',str(root/'models'),'--seed',str(a.seed),'--out',str(root/'readouts')])
        phase('collecting validation incidents',['microservice.collect','--split','test','--seed',str(a.test_seed),'--out',str(root/'test.jsonl')])
        phase('evaluating',['microservice.validate','evaluate','--data',str(root/'test.jsonl'),'--models',str(root/'models'),'--seed',str(a.seed),'--readouts',str(root/'readouts'),'--out',str(root/'evaluation')])
        (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
    finally:
        from microservice.live import reset
        reset()
if __name__=='__main__':main()
