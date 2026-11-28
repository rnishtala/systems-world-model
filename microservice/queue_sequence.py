"""Train on successive measured error windows, then roll the frozen prior forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from microservice.action_forecast import Forecaster, checkpoint_hash, verify_source
from microservice.action_outcomes import ridge_predict
from microservice.queue_drain import ORDERS
from microservice.queue_forecast import collect, encode, TARGETS, VERSION

SEQUENCE_VERSION = 'queue-drain-sequence-v1'


def recovery(pred):
    return any(a < .1 and b < .1 for a,b in zip(pred,pred[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if os.environ.get('POD_NAMESPACE') != 'systems-world-model-lab':
        raise ValueError('Wrong namespace')
    out,source = args.out,args.source
    if args.train:
        from microservice.configuration import elements
        from microservice.action_masking import masked_objective
        from microservice.action_train import train_partition
        with masked_objective():
            train_partition(out,json.loads((out/'manifest.json').read_text()),args.seed,25000,sequential=True)
        return
    meta = verify_source(source,VERSION,TARGETS)
    baseline = json.loads((source/'frozen.json').read_text())['ridge']
    seed = json.loads((source/'training-summary.json').read_text())['seed']
    out.mkdir(parents=True,exist_ok=False)
    (out/'train.jsonl').write_bytes((source/'train.jsonl').read_bytes())
    meta.update(version=SEQUENCE_VERSION,source_run=source.parent.name)
    (out/'manifest.json').write_text(json.dumps(meta))
    cases=[(ms,r,o) for ms in (91,94,96,100) for r in range(2) for o in ORDERS]
    random.Random(args.seed).shuffle(cases)
    (out/'protocol.json').write_text(json.dumps(dict(cases=cases,training_seed=seed,test_seed=args.seed,
        decision='More true recovery calls than frozen telemetry history and no more false recovery calls',
        predicted='Two consecutive predicted error fractions below 0.10',observed='Two consecutive zero-failure windows by eight seconds',
        rollout='Four two-second latent prior steps, load reduction then three no-op actions, no post-action observations')))
    subprocess.run([sys.executable,'-m','microservice.queue_sequence','--source',str(source),
                    '--out',str(out),'--seed',str(seed),'--train'],check=True)
    training=json.loads((out/'summary.json').read_text())
    (out/'training-summary.json').write_text(json.dumps(training))
    model=Forecaster(out,version=SEQUENCE_VERSION,targets=TARGETS,sequential=True)
    frozen=dict(checkpoint=checkpoint_hash(out),ridge=baseline)
    (out/'frozen.json').write_text(json.dumps(frozen))
    reports={m:dict(true_recovery=0,false_recovery=0,errors=[]) for m in ('dreamer','telemetry')}
    recovered=0
    for i,(ms,rep,order) in enumerate(cases):
        result,h,actual=collect(ms,order,out,str(i))
        preds=dict(dreamer=np.clip(model(h,'load'),0,1),telemetry=np.clip(ridge_predict(baseline,encode(h)),0,1))
        confirmed=actual['recovery'] is not None
        recovered+=int(confirmed)
        for name,pred in preds.items():
            report=reports[name]
            report['true_recovery' if confirmed else 'false_recovery']+=int(recovery(pred))
            report['errors'].append(float(np.mean(abs(pred-actual['fractions']))))
        (out/f'incident-{i}.json').write_text(json.dumps(dict(**result,case=cases[i],stats=actual,
                    predictions={k:v.tolist() for k,v in preds.items()})))
    for r in reports.values():
        r['mae']=float(np.mean(r.pop('errors')))
        r.update(recovered=recovered,unrecovered=len(cases)-recovered)
    if checkpoint_hash(out)!=frozen['checkpoint'] or hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest()!=meta['training_sha256']:
        raise ValueError('Frozen artifact changed')
    d,t=reports['dreamer'],reports['telemetry']
    passed=0<recovered<len(cases) and d['true_recovery']>t['true_recovery'] and d['false_recovery']<=t['false_recovery']
    (out/'summary.json').write_text(json.dumps(dict(queue_forecast=reports,recovery_decision=passed,action_dreamer=training,incidents=[])))

if __name__=='__main__':
    main()
