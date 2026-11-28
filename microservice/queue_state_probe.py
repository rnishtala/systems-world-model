"""Measure linear readability along the frozen sequential Dreamer rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from microservice.action_forecast import Forecaster, RNG_SEED, checkpoint_hash, prefix_frames, prior_outcomes
from microservice.action_outcomes import ACTIONS, PENALTIES, fit_ridge, ridge_predict
from microservice.queue_drain import ORDERS
from microservice.queue_forecast import collect, encode, time_rule, TARGETS
from microservice.queue_sequence import recovery, SEQUENCE_VERSION

STAGES = ('observed','step1','step2','step3','step4')


def extractor(model):
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    @nj.pure
    def run(h,action):
        return prior_outcomes(model.agent.model,h,action,jnp,sequential=True,state_path=True)
    compiled=jax.jit(lambda params,h,action,key:run(params,h,action,seed=key,create=False,modify=False)[1])
    action=2*np.eye(len(ACTIONS),dtype=np.float32)[ACTIONS.index('load')]-1
    def extract(h):
        with jax.transfer_guard('allow'):
            _,finite,states=compiled(model.params,prefix_frames(h),action,jax.random.PRNGKey(RNG_SEED))
            states=np.asarray(states)
        if not bool(finite) or len(states)!=5 or not np.isfinite(states).all():
            raise ValueError('Invalid frozen state path')
        return dict(telemetry_probe=encode(h),**dict(zip(STAGES,states)))
    return extract


def query_ms(h):
    value=time_rule(h)*1000
    if value<=0:raise ValueError('Query duration was not observed')
    return value


def fit(x,y,groups):
    query_options=[]; call_options=[]
    for p in PENALTIES:
        held=np.empty_like(y)
        for group in np.unique(groups):
            test=groups==group
            held[test]=ridge_predict(fit_ridge(x[~test],y[~test],p),x[test])
        query_options.append((np.abs(held[:,0]-y[:,0]).mean(),p))
        # Highest training recall at zero out-of-fold false calls. Break ties by
        # squared recovery-score error, then choose the more conservative cutoff.
        cutoff=np.nextafter(held[y[:,1]==0,1].max(),np.inf)
        tp=int(((held[:,1]>=cutoff)&(y[:,1]==1)).sum())
        call_options.append((-tp,np.mean((held[:,1]-y[:,1])**2),p,cutoff))
    _,qp=min(query_options)
    rank,mse,rp,cutoff=min(call_options)
    return dict(query=fit_ridge(x,y[:,0],qp),recover=fit_ridge(x,y[:,1],rp),
                cutoff=float(cutoff),selection=[rank,float(mse)],penalties=[qp,rp])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    src,out=args.source,args.out
    read=lambda path:json.loads(path.read_text())
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    model=Forecaster(src,SEQUENCE_VERSION,TARGETS,sequential=True)
    source_frozen=read(src/'frozen.json')
    if model.checkpoint_sha256!=source_frozen['checkpoint']:raise ValueError('Checkpoint changed')
    extract=extractor(model)
    rows=[json.loads(l) for l in (src/'train.jsonl').read_text().splitlines()]
    if len(rows)!=24 or any(r['split']!='train' or not r['restored'] for r in rows):raise ValueError('Invalid training rows')
    features=[extract(r['observations']) for r in rows]
    y=np.asarray([[query_ms(r['observations']),float(any(a==b==0 for a,b in zip(r['outcome'],r['outcome'][1:])))] for r in rows])
    groups=np.array([r['group'] for r in rows])
    probes={k:fit(np.stack([v[k] for v in features]),y,groups) for k in features[0]}
    primary=min(STAGES,key=lambda k:probes[k]['selection'])
    out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-probes.json'
    frozen.write_text(json.dumps(dict(probes=probes,primary=primary,ridge=source_frozen['ridge']),allow_nan=False))
    frozen_hash=digest(frozen)
    cases=[(ms,r,o) for ms in (91,94,96,100) for r in range(2) for o in ORDERS]
    random.Random(args.seed).shuffle(cases)
    (out/'protocol.json').write_text(json.dumps(dict(cases=cases,primary=primary,checkpoint=model.checkpoint_sha256,
        frozen=frozen_hash,criterion='More true calls than frozen telemetry history with no additional false calls',
        query_target='Query-count-weighted mean pre-action ledger query milliseconds',horizon_seconds=8)))
    records=[]
    for i,(ms,rep,order) in enumerate(cases):
        result,h,actual=collect(ms,order,out,str(i))
        vec=extract(h)
        pred={}
        for k,m in probes.items():
            score=float(ridge_predict(m['recover'],vec[k]))
            pred[k]=dict(query=float(ridge_predict(m['query'],vec[k])),score=score,call=score>=m['cutoff'])
        hist=np.clip(ridge_predict(source_frozen['ridge'],vec['telemetry_probe']),0,1)
        pred['telemetry_history']=dict(call=recovery(hist))
        record=dict(case=cases[i],query=query_ms(h),recovered=actual['recovery'] is not None,pred=pred)
        records.append(record)
        (out/f'incident-{i}.json').write_text(json.dumps(dict(**result,probe=record)))
    metrics={}
    for k in records[0]['pred']:
        tp=sum(int(r['recovered'] and r['pred'][k]['call']) for r in records)
        fp=sum(int(not r['recovered'] and r['pred'][k]['call']) for r in records)
        m=metrics[k]=dict(true_recovery=tp,false_recovery=fp)
        if k in probes:
            m['query_mae_ms']=float(np.mean([abs(r['query']-r['pred'][k]['query']) for r in records]))
            pos=[r['pred'][k]['score'] for r in records if r['recovered']]
            neg=[r['pred'][k]['score'] for r in records if not r['recovered']]
            m['auc']=float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg])) if pos and neg else None
    if digest(frozen)!=frozen_hash or checkpoint_hash(src)!=model.checkpoint_sha256:raise ValueError('Frozen fit changed')
    a,b=metrics[primary],metrics['telemetry_history']
    n=sum(r['recovered'] for r in records)
    summary=dict(metrics=metrics,recovered=n,unrecovered=len(records)-n,primary=primary,
                 constant_query_mae_ms=float(np.mean([abs(r['query']-np.median(y[:,0])) for r in records])),
                 passes=0<n<len(records) and a['true_recovery']>b['true_recovery'] and a['false_recovery']<=b['false_recovery'])
    (out/'summary.json').write_text(json.dumps(dict(queue_state_probe=summary,incidents=[])))


if __name__=='__main__':
    main()
