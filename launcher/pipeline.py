"""Immutable per-run process: train, freeze, collect fresh test, evaluate."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from launcher.common import dataset_check,NAMESPACE,validate_expansion

child=None

def stop(signum,frame):
    if child is not None and child.poll() is None:child.terminate()
    raise SystemExit(128+signum)

def main():
    global child
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--test-seed',type=int,required=True);p.add_argument('--dataset-sha256');p.add_argument('--expand-training-seed',type=int);p.add_argument('--baseline-run');p.add_argument('--training-source-run');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE:raise RuntimeError('Wrong namespace')
    import re
    if not re.fullmatch('[a-f0-9]{12}',a.run):raise ValueError('Invalid run ID')
    root=Path('/results/experiments/runs')/a.run;root.mkdir(parents=True,exist_ok=True)
    if (root/'models').exists():raise RuntimeError('Refusing to overwrite an existing training attempt')
    data=dataset_check(a.dataset,a.dataset_sha256)
    if (a.expand_training_seed is None)!=(a.baseline_run is None):raise ValueError('Expansion and baseline must be supplied together')
    if a.expand_training_seed is not None:validate_expansion(a.expand_training_seed,a.baseline_run,a.test_seed)
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
        if a.training_source_run:
            from launcher.common import validate_retest
            from launcher.retest import frozen_identity
            import hashlib
            validate_retest(a.run,a.training_source_run)
            if a.expand_training_seed is not None or a.baseline_run:raise ValueError('Training comparison must run alone')
            base=root.parent/a.training_source_run
            frozen=frozen_identity(base);baseline_seed=frozen['seed']
            if baseline_seed!=a.seed or frozen['steps']!=a.steps:raise ValueError('Keep model seed and steps fixed')
            source=base/'train.jsonl'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=frozen['train_sha256']:raise ValueError('Source training data changed')
            used=set(frozen['training_seeds'])
            for path in root.parent.glob('*/test.jsonl'):
                used.update(json.loads(line)['seed'] for line in path.read_text().splitlines())
            if a.test_seed in used:raise ValueError('Test seed already used')
            data=root/'train.jsonl';data.write_bytes(source.read_bytes())
            a.baseline_run=a.training_source_run
            protocol=dict(change='Supervise diagnosis at six and eight windows instead of eight only',supervision_windows=[6,8],baseline_run=a.baseline_run,baseline_identity=frozen,model_seed=a.seed,steps=a.steps,training_sha256=frozen['train_sha256'],test_seed=a.test_seed,utc=time.time())
            (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        if a.expand_training_seed is not None:
            import hashlib
            from microservice.validate import read_rows,load_readouts
            from microservice.readout import digest
            from launcher.datasets import validate_jsonl,Catalog
            base=Path('/results/experiments/runs')/a.baseline_run
            # The baseline is the completed standard run; its training metadata
            # and frozen artifacts establish the actual comparison identity.
            candidates=list((base/'models').glob('training-*.json'))
            if len(candidates)!=1:raise ValueError('Expected exactly one baseline training identity')
            metadata=json.loads(candidates[0].read_text());baseline_seed=metadata['seed']
            frozen,_=load_readouts(base/'readouts')
            if metadata['seed']!=a.seed or metadata['steps']!=a.steps:raise ValueError('Keep baseline model seed and step budget fixed')
            original=read_rows(data)
            if frozen['train_sha256']!=hashlib.sha256(data.read_bytes()).hexdigest():raise ValueError('Expansion must start with the baseline training data')
            if digest(base/'models'/f'final-{baseline_seed}')!=frozen['checkpoint_sha256']:raise ValueError('Baseline checkpoint changed')
            used={r['seed'] for r in original}
            for path in Path('/results/experiments/runs').glob('*/test.jsonl'):
                used.update(r['seed'] for r in read_rows(path))
            if a.expand_training_seed in used or a.test_seed in used:raise ValueError('Collection seed already used')
            protocol=dict(base_dataset=a.dataset,base_sha256=frozen['train_sha256'],baseline_run=a.baseline_run,baseline_checkpoint_sha256=frozen['checkpoint_sha256'],baseline_readouts_sha256=frozen['readouts_sha256'],model_seed=a.seed,steps=a.steps,expansion_seed=a.expand_training_seed,test_seed=a.test_seed,base_incidents=len(original),added_incidents=64,utc=time.time(),change='Append fresh balanced training incidents; model and training settings unchanged')
            (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
            phase('collecting additional training incidents',['microservice.collect','--split','train','--seed',str(a.expand_training_seed),'--out',str(root/'additional-train.jsonl')])
            added_text=(root/'additional-train.jsonl').read_text()
            validate_jsonl(added_text,Catalog().heldout())
            # Preserve historical split annotations byte-for-byte. Only the
            # explicitly designated baseline training file is extended.
            expanded=root/'train.jsonl'
            expanded.write_bytes(data.read_bytes().rstrip(b'\n')+b'\n'+added_text.encode())
            data=expanded
            protocol.update(training_incidents=len(read_rows(data)),training_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
            (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        phase('training',['microservice.train','--data',str(data),'--out',str(root/'models'),'--seed',str(a.seed),'--steps',str(a.steps)])
        phase('fitting readouts',['microservice.validate','fit','--data',str(data),'--models',str(root/'models'),'--seed',str(a.seed),'--out',str(root/'readouts')])
        phase('collecting validation incidents',['microservice.collect','--split','test','--seed',str(a.test_seed),'--out',str(root/'test.jsonl')])
        if a.baseline_run:
            phase('evaluating frozen baseline',['microservice.validate','evaluate','--data',str(root/'test.jsonl'),'--models',str(base/'models'),'--seed',str(baseline_seed),'--readouts',str(base/'readouts'),'--out',str(root/'baseline-evaluation')])
        phase('evaluating',['microservice.validate','evaluate','--data',str(root/'test.jsonl'),'--models',str(root/'models'),'--seed',str(a.seed),'--readouts',str(root/'readouts'),'--out',str(root/'evaluation')]+(['--baseline',str(root/'baseline-evaluation')] if a.baseline_run else []))
        if a.training_source_run and frozen_identity(base)!=frozen:raise ValueError('Baseline changed during comparison')
        (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
    finally:
        from microservice.live import reset
        reset()
if __name__=='__main__':main()
