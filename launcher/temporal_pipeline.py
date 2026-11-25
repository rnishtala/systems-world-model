"""Run the bounded temporal mechanism pilot through the existing lab queue."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from launcher import pipeline
from launcher.common import NAMESPACE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset'):p.add_argument('--'+name,required=True)
    for name in ('seed','steps','test-seed'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--action-comparison-source');p.add_argument('--dataset-sha256');p.add_argument('--action-inference-source');p.add_argument('--action-dreamer-source');p.add_argument('--action-balanced',action='store_true');p.add_argument('--action-source');p.add_argument('--action-validation-source');modes=p.add_mutually_exclusive_group();modes.add_argument('--snapshot-calibration',action='store_true');modes.add_argument('--action-outcomes',action='store_true');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE:raise ValueError('Wrong namespace')
    resume=[]
    if a.action_balanced:
        if not a.action_outcomes or a.action_source or a.action_validation_source:raise ValueError('Balanced collection requires fresh action outcomes')
        resume=['--balanced']
    if a.action_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_source)
        if not a.action_outcomes:raise ValueError('Recovery requires action outcomes')
        resume=['--resume-from',str(Path('/results/experiments/runs')/a.action_source/'temporal-pilot')]
    if a.action_validation_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_validation_source)
        if not a.action_outcomes or a.action_source:raise ValueError('Validation requires action outcomes without recovery')
        resume=['--validate-from',str(Path('/results/experiments/runs')/a.action_validation_source/'temporal-pilot')]
    if a.action_comparison_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_comparison_source)
        if a.action_inference_source or a.action_dreamer_source or a.action_outcomes or a.snapshot_calibration or resume:raise ValueError('Action comparison must run alone')
    if a.action_inference_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_inference_source)
        if a.action_dreamer_source or a.action_outcomes or a.snapshot_calibration or resume:raise ValueError('Action inference must run alone')
    if a.action_dreamer_source:
        from launcher.common import validate_retest
        validate_retest(a.run,a.action_dreamer_source)
        if a.action_outcomes or a.snapshot_calibration or resume or a.steps!=1000:raise ValueError('Action Dreamer smoke must run alone with 1000 steps')
    root=Path('/results/experiments/runs')/a.run;root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Use a fresh pilot run')
    (root/'protocol.json').write_text(json.dumps(dict(mode='temporal-pilot',seed=a.seed if a.action_dreamer_source else a.test_seed,action_dreamer_source=a.action_dreamer_source,scope='Fresh frozen action outcome comparison' if a.action_comparison_source else 'Action Dreamer inference smoke only; no accuracy evaluation' if a.action_inference_source else 'Action Dreamer training smoke only; no evaluation' if a.action_dreamer_source else 'Mechanism feasibility only; no Dreamer training',utc=time.time())))
    (root/'progress.json').write_text(json.dumps(dict(phase='comparing frozen Dreamer on fresh incidents' if a.action_comparison_source else 'verifying action-conditioned inference' if a.action_inference_source else 'training action-conditioned Dreamer' if a.action_dreamer_source else 'collecting temporal mechanism pilot',utc=time.time())))
    signal.signal(signal.SIGTERM,pipeline.stop);signal.signal(signal.SIGINT,pipeline.stop)
    with (root/'run.log').open('ab',buffering=0) as log:
        command=[sys.executable,'-m',('microservice.action_outcomes' if a.action_outcomes else 'microservice.temporal_calibration' if a.snapshot_calibration else 'microservice.temporal_pilot'),'--out',str(root/'temporal-pilot'),'--seed',str(a.test_seed)]+resume
        if a.action_dreamer_source:command=[sys.executable,'-m','microservice.action_train','--source',str(Path('/results/experiments/runs')/a.action_dreamer_source/'temporal-pilot'),'--out',str(root/'temporal-pilot'),'--seed',str(a.seed),'--steps',str(a.steps)]
        if a.action_inference_source:command=[sys.executable,'-m','microservice.action_forecast','--source',str(Path('/results/experiments/runs')/a.action_inference_source/'temporal-pilot'),'--out',str(root/'temporal-pilot')]
        if a.action_comparison_source:command=[sys.executable,'-m','microservice.action_compare','--source',str(Path('/results/experiments/runs')/a.action_comparison_source/'temporal-pilot'),'--out',str(root/'temporal-pilot'),'--seed',str(a.test_seed)]
        pipeline.child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        if pipeline.child.wait():raise RuntimeError('Temporal pilot failed; inspect restoration status')
    if a.action_dreamer_source:
        path=root/'temporal-pilot'/'summary.json';result=json.loads(path.read_text());path.write_text(json.dumps(dict(action_dreamer=result,incidents=[])))
    if a.action_inference_source:
        path=root/'temporal-pilot'/'summary.json';result=json.loads(path.read_text());path.write_text(json.dumps(dict(action_inference=result,incidents=[])))
    (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))

if __name__=='__main__':main()
