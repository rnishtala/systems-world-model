"""Prospective paired confirmation of the previously frozen step-two recovery probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from microservice.action_forecast import Forecaster, checkpoint_hash
from microservice.action_outcomes import ridge_predict
from microservice.queue_drain import ORDERS
from microservice.queue_forecast import collect, TARGETS
from microservice.queue_sequence import SEQUENCE_VERSION, recovery
from microservice.queue_state_probe import extractor, query_ms


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if os.environ.get('POD_NAMESPACE') != 'systems-world-model-lab':
        raise ValueError('Wrong namespace')
    source, out = args.source, args.out
    protocol = read(source/'protocol.json')
    # The discovery batch identified the candidate; pin the exact fitted artifact.
    raw = (source/'frozen-probes.json').read_bytes()
    probe_hash = hashlib.sha256(raw).hexdigest()
    if probe_hash != protocol['frozen'] or probe_hash != '178d830a1624ab472673b196a14e0b473aa8f99fdcc8f2498778691c7b408d11':
        raise ValueError('Unexpected discovery fit')
    frozen = json.loads(raw)
    candidate = frozen['probes']['step2']
    runs = source.parent.parent
    seq = runs/'19daefdcdadb'/'temporal-pilot'
    original = read(runs/'40443d020ad9'/'temporal-pilot'/'frozen.json')
    if frozen['ridge'] != original['ridge'] or original['cutoff'] != .0953273:
        raise ValueError('Telemetry controls changed')
    model = Forecaster(seq, SEQUENCE_VERSION, TARGETS, sequential=True)
    if model.checkpoint_sha256 != protocol['checkpoint']:
        raise ValueError('Unexpected checkpoint')
    extract = extractor(model)
    cases = [(ms,rep,order) for ms in (91,94,96,100) for rep in range(4) for order in ORDERS]
    random.Random(args.seed).shuffle(cases)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(seed=args.seed,cases=cases,primary='step2',
        frozen=probe_hash,checkpoint=model.checkpoint_sha256,query_cutoff_ms=1000*original['cutoff'],
        probe_threshold=candidate['cutoff'],source_run=source.parent.name,
        criterion='Strictly more true recoveries and no more false calls than each control by eight seconds')))
    metrics = {name:dict(true_recovery=0,false_recovery=0) for name in ('step2','telemetry_history','query_cutoff')}
    paired = {name:dict(recovered=[0]*4,unrecovered=[0]*4) for name in ('telemetry_history','query_cutoff')}
    n = 0
    for i,(ms,rep,order) in enumerate(cases):
        result,h,actual = collect(ms,order,out,str(i))
        features = extract(h)
        score = float(ridge_predict(candidate['recover'],features['step2']))
        ms_query = query_ms(h)
        control = np.clip(ridge_predict(frozen['ridge'],features['telemetry_probe']),0,1)
        calls = dict(step2=score>=candidate['cutoff'],telemetry_history=recovery(control),
                     query_cutoff=ms_query<1000*original['cutoff'])
        confirmed = actual['recovery'] is not None
        n += int(confirmed)
        for name,call in calls.items():
            metrics[name]['true_recovery' if confirmed else 'false_recovery'] += int(call)
        for name in paired:
            index = 2*int(calls['step2']) + int(calls[name])
            paired[name]['recovered' if confirmed else 'unrecovered'][index] += 1
        (out/f'incident-{i}.json').write_text(json.dumps(dict(**result,case=cases[i],stats=actual,
                    calls=calls,query_ms=ms_query,step2_score=score,telemetry_forecast=control.tolist())))
    if checkpoint_hash(seq)!=model.checkpoint_sha256 or (source/'frozen-probes.json').read_bytes()!=raw:
        raise ValueError('Frozen artifacts changed')
    d=metrics['step2']
    passed=0<n<len(cases) and all(d['true_recovery']>metrics[k]['true_recovery'] and
                                   d['false_recovery']<=metrics[k]['false_recovery'] for k in paired)
    summary=dict(metrics=metrics,primary='step2',recovered=n,unrecovered=len(cases)-n,passes=passed,
                 paired=paired,pair_order=['neither','control only','step2 only','both'],confirmation=True)
    (out/'summary.json').write_text(json.dumps(dict(queue_state_probe=summary,incidents=[])))


if __name__=='__main__':
    main()
