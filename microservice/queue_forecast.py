"""Forecast the next four error-rate windows after an observed load reduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from microservice.action_model import validate
from microservice.action_outcomes import ACTIONS, observations, fit_ridge, ridge_predict
from microservice.queue_drain import trajectory, ORDERS
from microservice.temporal_calibration import FEATURES
from microservice.temporal_pilot import incident

TARGETS = [f'checkout_errors_{t-2}_{t}s' for t in (2,4,6,8)]
VERSION = 'queue-drain-trajectory-v1'


def encode(history):
    h = np.array(history)
    # Ordinary telemetry control sees the identical history plus explicit changes.
    return np.r_[np.log1p(h).ravel(), np.diff(np.log1p(h),axis=0).ravel()]


def time_rule(history):
    h = np.asarray(history)
    count = h[:,FEATURES.index('ledger.db_query_count')]
    duration = h[:,FEATURES.index('ledger.db_query_mean')]
    return float(np.average(duration, weights=count)) if count.sum() else 0.


def train(out, meta, seed):
    from microservice.configuration import elements
    from microservice.action_masking import masked_objective
    from microservice.action_train import train_partition
    with masked_objective():
        train_partition(out, meta, seed, 25000)


def valid_measurement(result):
    if not result['restored']:
        raise RuntimeError('Lab restoration failed')
    return observations(result['windows']), trajectory(result)


def collect(ms, order, out, case):
    for attempt in range(3):
        result = incident('database', 'load', dict(fault_hold=ms/1000, healthy_hold=.04,
                          onset=1, relief_at=6, duration=16, base_rate=8, burst_order=order))
        try:
            history, stats = valid_measurement(result)
        except (ValueError, StopIteration):
            (out/f'incomplete-{case}-{attempt}.json').write_text(json.dumps(result))
            if attempt == 2:
                raise
            continue
        return result, history, stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--train',action='store_true')
    a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':
        raise ValueError('Wrong namespace')
    out=a.out
    if a.train:
        train(out,json.loads((out/'manifest.json').read_text()),a.seed)
        return
    source_summary=json.loads((a.source/'summary.json').read_text())['queue_drain']
    if not source_summary['passes']:
        raise ValueError('Required recovery variation absent')
    out.mkdir(parents=True,exist_ok=False)
    train_rows=[]
    for f in sorted(a.source.glob('incident-*.json')):
        r=json.loads(f.read_text())
        ms,repeat,order=r['case']
        try:
            history, stats = valid_measurement(r)
        except (ValueError, StopIteration):
            replacement, history, stats = collect(ms, order, out, 'train-'+f.stem)
            (out/f'replacement-{f.name}').write_text(json.dumps(dict(case=r['case'], **replacement)))
        train_rows.append(dict(split='train',action='load',group=str(ms),restored=r['restored'],
                observations=history,outcome=stats['fractions']))
    validate(train_rows)
    if len(train_rows)!=24:raise ValueError('Incomplete source')
    data=''.join(json.dumps(r)+'\n' for r in train_rows).encode()
    (out/'train.jsonl').write_bytes(data)
    meta=dict(version=VERSION,training_sha256=hashlib.sha256(data).hexdigest(),features=FEATURES,
              targets=TARGETS,actions=ACTIONS,source_run=a.source.parent.name,training_trials=len(train_rows))
    (out/'manifest.json').write_text(json.dumps(meta))
    cases=[(ms,i,o) for ms in (91,94,96,100) for i in range(2) for o in ORDERS]
    random.Random(a.seed).shuffle(cases)
    (out/'protocol.json').write_text(json.dumps(dict(cases=cases,seed=a.seed,primary='Mean absolute error across four bins per incident',false_recovery='Any two consecutive predicted fractions below 0.10 in an incident without confirmed recovery by eight seconds',training='Masked DreamerV3, 25000 steps, four outcome channels forecast together',decision='Lower MAE than both controls and no extra false recovery calls versus either')))
    subprocess.run([sys.executable,'-m','microservice.queue_forecast','--source',str(a.source),
                    '--out',str(out),'--seed',str(a.seed),'--train'],check=True)
    from microservice.action_forecast import Forecaster, checkpoint_hash
    training_summary=json.loads((out/'summary.json').read_text())
    (out/'training-summary.json').write_text(json.dumps(training_summary))
    model=Forecaster(out,version=VERSION,targets=TARGETS)
    checkpoint=checkpoint_hash(out)
    x=np.stack([encode(r['observations']) for r in train_rows])
    y=np.array([r['outcome'] for r in train_rows])
    groups=np.array([r['group'] for r in train_rows])
    def loss(penalty):
        errors=[]
        for g in set(groups):
            fit=groups!=g
            pred=ridge_predict(fit_ridge(x[fit],y[fit],penalty),x[~fit])
            errors.extend(abs(np.clip(pred,0,1)-y[~fit]).ravel())
        return np.mean(errors)
    penalty=min((1,10,100),key=loss)
    ridge=fit_ridge(x,y,penalty)
    (out/'frozen.json').write_text(json.dumps(dict(checkpoint=checkpoint,ridge=ridge,penalty=penalty,cutoff=.0953273)))
    rows=[]
    for i,(ms,rep,order) in enumerate(cases):
        result,h,actual = collect(ms, order, out, str(i))
        predictions = dict(dreamer=np.clip(model(h,'load'),0,1).tolist(),
                           telemetry=np.clip(ridge_predict(ridge,encode(h)),0,1).tolist(),
                           cutoff=[float(time_rule(h)>=.0953273)]*4)
        rows.append(dict(case=cases[i],stats=actual,predictions=predictions))
        (out/f'incident-{i}.json').write_text(json.dumps(dict(**result,**rows[-1])))
    scores={}
    for method in ('dreamer','telemetry','cutoff'):
        errors=[];false=0
        for r in rows:
            pred=np.array(r['predictions'][method])
            errors.append(abs(pred-r['stats']['fractions']))
            false+=int(r['stats']['recovery'] is None and any((pred[:-1]<.1)&(pred[1:]<.1)))
        scores[method]=dict(mae=float(np.mean(errors)),false_recovery=false)
    if checkpoint_hash(out)!=checkpoint:raise ValueError('Checkpoint changed')
    (out/'summary.json').write_text(json.dumps(dict(queue_forecast=scores,action_dreamer=training_summary,incidents=[])))

if __name__=='__main__':main()
