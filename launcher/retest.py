"""Evaluate two frozen models on a new shared batch, without training or fitting."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from launcher.common import NAMESPACE,validate_retest
from launcher.pipeline import stop
from launcher import pipeline


def frozen_identity(root):
    from microservice.validate import load_readouts
    from microservice.readout import digest
    files=list((root/'models').glob('training-*.json'))
    if len(files)!=1:raise ValueError('Expected one training identity')
    metadata=json.loads(files[0].read_text());frozen,_=load_readouts(root/'readouts')
    if digest(root/'models'/f"final-{metadata['seed']}")!=frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    if metadata['calibration_sha256']!=frozen['train_sha256']:raise ValueError('Training identity mismatch')
    return dict(seed=metadata['seed'],steps=metadata['steps'],**frozen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset','candidate-run','baseline-run'):p.add_argument('--'+name,required=True)
    for name in ('seed','steps','test-seed'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--dataset-sha256');a=p.parse_args()
    validate_retest(a.candidate_run,a.baseline_run)
    import re
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE or not re.fullmatch('[a-f0-9]{12}',a.run):raise ValueError('Invalid lab run')
    root=Path('/results/experiments/runs')/a.run
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() or (root/'test.jsonl').exists():raise ValueError('Use a fresh test run')
    candidate=root.parent/a.candidate_run;baseline=root.parent/a.baseline_run
    identities={name:frozen_identity(path) for name,path in [('candidate',candidate),('baseline',baseline)]}
    if identities['candidate']['seed']!=a.seed or identities['candidate']['steps']!=a.steps:raise ValueError('Candidate settings mismatch')
    used=set().union(*(set(v['training_seeds']) for v in identities.values()))
    for path in root.parent.glob('*/test.jsonl'):
        used.update(json.loads(line)['seed'] for line in path.read_text().splitlines())
    if a.test_seed in used:raise ValueError('Test seed already used')
    protocol=dict(mode='test-only',candidate_run=a.candidate_run,baseline_run=a.baseline_run,test_seed=a.test_seed,identities=identities,utc=time.time())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def phase(name,args):
        (root/'progress.json').write_text(json.dumps(dict(phase=name,utc=time.time())))
        print('EXPERIMENT_PHASE',name,flush=True)
        with (root/'run.log').open('ab',buffering=0) as log:
            pipeline.child=subprocess.Popen([sys.executable,'-m',*args],stdout=log,stderr=subprocess.STDOUT)
            code=pipeline.child.wait()
        if code:raise RuntimeError(f'{name} exited with status {code}')
    try:
        phase('collecting validation incidents',['microservice.collect','--split','test','--seed',str(a.test_seed),'--out',str(root/'test.jsonl')])
        for name,source,out in [('baseline',baseline,'baseline-evaluation'),('candidate',candidate,'evaluation')]:
            args=['microservice.validate','evaluate','--data',str(root/'test.jsonl'),'--models',str(source/'models'),'--seed',str(identities[name]['seed']),'--readouts',str(source/'readouts'),'--out',str(root/out)]
            if name=='candidate':args+=['--baseline',str(root/'baseline-evaluation')]
            phase('evaluating frozen '+name,args)
        for name,path in [('candidate',candidate),('baseline',baseline)]:
            if frozen_identity(path)!=identities[name]:raise ValueError('Frozen artifacts changed during test')
        (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
    finally:
        from microservice.live import reset
        reset()

if __name__=='__main__':main()
