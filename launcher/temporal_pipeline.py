"""Run the bounded temporal mechanism pilot through the existing lab queue."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from launcher import pipeline
from launcher.common import NAMESPACE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset'):p.add_argument('--'+name,required=True)
    for name in ('seed','steps','test-seed'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--dataset-sha256');p.add_argument('--action-source');modes=p.add_mutually_exclusive_group();modes.add_argument('--snapshot-calibration',action='store_true');modes.add_argument('--action-outcomes',action='store_true');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE:raise ValueError('Wrong namespace')
    resume=[]
    if a.action_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_source)
        if not a.action_outcomes:raise ValueError('Recovery requires action outcomes')
        resume=['--resume-from',str(Path('/results/experiments/runs')/a.action_source/'temporal-pilot')]
    root=Path('/results/experiments/runs')/a.run;root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Use a fresh pilot run')
    (root/'protocol.json').write_text(json.dumps(dict(mode='temporal-pilot',seed=a.test_seed,scope='Mechanism feasibility only; no Dreamer training',utc=time.time())))
    (root/'progress.json').write_text(json.dumps(dict(phase='collecting temporal mechanism pilot',utc=time.time())))
    signal.signal(signal.SIGTERM,pipeline.stop);signal.signal(signal.SIGINT,pipeline.stop)
    with (root/'run.log').open('ab',buffering=0) as log:
        pipeline.child=subprocess.Popen([sys.executable,'-m',('microservice.action_outcomes' if a.action_outcomes else 'microservice.temporal_calibration' if a.snapshot_calibration else 'microservice.temporal_pilot'),'--out',str(root/'temporal-pilot'),'--seed',str(a.test_seed)]+resume,stdout=log,stderr=subprocess.STDOUT)
        if pipeline.child.wait():raise RuntimeError('Temporal pilot failed; inspect restoration status')
    (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))

if __name__=='__main__':main()
