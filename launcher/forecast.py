"""Fit controls on source training data, freeze, then acquire fresh forecast tests."""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from launcher.common import NAMESPACE,validate_retest
from launcher.retest import frozen_identity
from launcher import pipeline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dataset','candidate-run'):p.add_argument('--'+name,required=True)
    for name in ('seed','steps','test-seed'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--fresh-triage',action='store_true');p.add_argument('--triage',action='store_true');p.add_argument('--audit-source');p.add_argument('--dataset-sha256');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!=NAMESPACE:raise ValueError('Wrong namespace')
    validate_retest(a.run,a.candidate_run)
    root=Path('/results/experiments/runs')/a.run;source=root.parent/a.candidate_run
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Use a new forecast run')
    identity=frozen_identity(source)
    if identity['seed']!=a.seed or identity['steps']!=a.steps:raise ValueError('Source settings mismatch')
    from microservice.validate import sha
    data=source/'train.jsonl'
    if sha(data)!=identity['train_sha256']:raise ValueError('Source data changed')
    used=set(identity['training_seeds'])
    for path in root.parent.glob('*/test.jsonl'):used.update(json.loads(line)['seed'] for line in path.read_text().splitlines())
    if a.test_seed in used:raise ValueError('Test seed already used')
    schedule=json.loads((source/'protocol.json').read_text()).get('probe_schedule','mixed')
    if schedule not in ('mixed','ledger-first'):raise ValueError('Invalid source schedule')
    protocol=dict(mode='forecast',candidate_run=a.candidate_run,source_identity=identity,probe_schedule=schedule,test_seed=a.test_seed,utc=time.time(),cuts=[4,5,6,7],target='Next-window ledger log latency',controls=['persistence','matching_probe','regression'],comparison_scope='Same history and known next probe; no next telemetry in Dreamer prior',success='At least 10% lower log MAE than training-selected control and paired incident bootstrap upper bound below zero')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    signal.signal(signal.SIGTERM,pipeline.stop);signal.signal(signal.SIGINT,pipeline.stop)
    def phase(name,args):
        (root/'progress.json').write_text(json.dumps(dict(phase=name,utc=time.time())))
        print('EXPERIMENT_PHASE',name,flush=True)
        with (root/'run.log').open('ab',buffering=0) as log:
            pipeline.child=subprocess.Popen([sys.executable,'-m',*args],stdout=log,stderr=subprocess.STDOUT)
            code=pipeline.child.wait()
        if code:raise RuntimeError(f'{name} exited with status {code}')
    common=['--models',str(source/'models'),'--seed',str(a.seed)]
    if a.fresh_triage:
        if a.audit_source:raise ValueError('Fresh triage cannot reuse tests')
        protocol.update(mode='fresh-triage',primary_prefix=6,scope='Freeze readouts on source training data, then collect fresh incidents',success='Report paired exact diagnosis accuracy and false pool counts at six observations; other prefixes secondary')
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        try:
            phase('freezing triage readouts',['microservice.triage_validate','fit','--data',str(data),'--out',str(root/'triage-readouts'),*common])
            phase('collecting validation incidents',['microservice.collect','--split','test','--seed',str(a.test_seed),'--out',str(root/'test.jsonl'),'--probe-schedule',schedule])
            phase('evaluating frozen triage readouts',['microservice.triage_validate','evaluate','--data',str(root/'test.jsonl'),'--out',str(root/'forecast-diagnostic'),'--readouts',str(root/'triage-readouts'),*common])
            if frozen_identity(source)!=identity:raise ValueError('Frozen source artifacts changed')
            (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
        finally:
            from microservice.live import reset
            reset()
        return
    if a.audit_source:
        validate_retest(a.run,a.audit_source)
        retained=root.parent/a.audit_source
        previous=json.loads((retained/'protocol.json').read_text())
        if previous['candidate_run']!=a.candidate_run:raise ValueError('Diagnostic source mismatch')
        if not (retained/'forecast-evaluation/summary.json').exists():raise ValueError('Forecast must have completed')
        protocol.update(mode='forecast-diagnostic',audit_source=a.audit_source,scope='Existing training and retained test only; no fitting or fault collection')
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        phase('auditing residual-assisted triage' if a.triage else 'auditing posterior reconstruction and prior forecasts',['microservice.triage_audit' if a.triage else 'microservice.forecast_audit','--train',str(data),'--test',str(retained/'test.jsonl'),'--original-predictions',str(retained/'forecast-evaluation/predictions.json'),'--out',str(root/'forecast-diagnostic'),*common])
        if frozen_identity(source)!=identity:raise ValueError('Frozen source artifacts changed')
        (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
        return
    try:
        phase('freezing forecast controls and checking Dreamer prior',['microservice.forecast','prepare','--data',str(data),'--out',str(root/'forecast-controls'),*common])
        phase('collecting fresh forecast incidents',['microservice.collect','--split','test','--seed',str(a.test_seed),'--out',str(root/'test.jsonl'),'--probe-schedule',schedule])
        phase('evaluating next-observation forecasts',['microservice.forecast','evaluate','--data',str(root/'test.jsonl'),'--out',str(root/'forecast-evaluation'),'--readouts',str(root/'forecast-controls'),*common])
        if frozen_identity(source)!=identity:raise ValueError('Frozen source artifacts changed')
        (root/'progress.json').write_text(json.dumps(dict(phase='completed',utc=time.time())))
    finally:
        from microservice.live import reset
        reset()

if __name__=='__main__':main()
