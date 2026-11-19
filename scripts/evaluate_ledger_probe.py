"""Frozen, matched-budget classifiers for the direct ledger pilot."""
import json,hashlib,sys,time,math
from pathlib import Path
import numpy as np
from microservice.load_response import response
from microservice.latent_probe import labels
from microservice.partial_readout import fit,score,diagnose
from scripts.develop_forecast_readout import choose
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/ledger-probe-v1'

def features(rows):
    base=np.array([response({'frames':r['base']}) for r in rows]);result={'base':base}
    for target in ('checkout','ledger'):
        extra=[]
        for row in rows:
            a,b=sorted(row['extra'][target],key=lambda f:f['requests'])
            low,high=np.log1p(np.array([a['metrics']['ledger']['latency_p95'],b['metrics']['ledger']['latency_p95']])/.1)
            extra.append([low,high,(high-low)/(b['requests']-a['requests'])])
        result[target]=np.concatenate([base,np.array(extra)],axis=1)
    return result

def rows(split):return [json.loads(l) for l in (O/f'{split}.jsonl').read_text().splitlines()]

def metrics(pred,rs):
    target=diagnose(labels(rs));pd=(pred&6)==(target&6);exact=pred==target
    return dict(pool_delay_correct=int(sum(pd)),exact_correct=int(sum(exact)),
        bit_errors=np.sum(((pred[:,None]>>np.arange(3))&1)!=((target[:,None]>>np.arange(3))&1),axis=0).tolist(),
        families={str(f):dict(n=4,pool_delay_correct=int(sum(pd[i] for i,r in enumerate(rs) if r['fault']==f)),exact_correct=int(sum(exact[i] for i,r in enumerate(rs) if r['fault']==f))) for f in range(8)},
        order={o:dict(n=sum(r['order'][0]==o for r in rs),pool_delay_correct=int(sum(pd[i] for i,r in enumerate(rs) if r['order'][0]==o))) for o in ('checkout','ledger')})

def freeze():
    rs=rows('development');xs=features(rs);y=labels(rs);groups=np.array([r['repeat'] for r in rs]);models={};selection={}
    for name,x in xs.items():
        config,counts=choose(x,y,groups);models[name]=fit(x,y,config);selection[name]=dict(config=config,cv_exact_correct=counts)
    np.savez_compressed(O/'readouts.npz',**{f'{n}_{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
    paths=[O/'development.jsonl',O/'readouts.npz',R/'reports/LEDGER_PROBE_PROTOCOL.md',Path(__file__),R/'microservice/ledger_probe.py']
    data=dict(utc=time.time(),selection=selection,hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (O/'frozen.json').write_text(json.dumps(data,indent=2)+'\n');print('READOUTS_FROZEN',flush=True)

def evaluate():
    frozen=json.loads((O/'frozen.json').read_text())
    for p,h in frozen['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    rs=rows('test');assert len(rs)==32 and all(r['utc']>frozen['utc'] for r in rs)
    xs=features(rs);arrays=np.load(O/'readouts.npz');preds={}
    for name,x in xs.items():
        m={k:arrays[f'{name}_{k}'] for k in ('mean','scale','bias','z','dual')};m['config']=frozen['selection'][name]['config']
        preds[name]=diagnose(score(m,x))
    result={n:metrics(p,rs) for n,p in preds.items()};a=result['ledger'];b=result['checkout']
    gate=bool(a['pool_delay_correct']>=28 and a['pool_delay_correct']>=b['pool_delay_correct']+4 and a['families']['0']['pool_delay_correct']==4 and all(v['pool_delay_correct']>=3 for v in a['families'].values()))
    target=diagnose(labels(rs));la=(preds['ledger']&6)==(target&6);co=(preds['checkout']&6)==(target&6)
    wins=int(sum(la&~co));losses=int(sum(~la&co));n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    summary=dict(methods=result,screening_gate=gate,paired=dict(ledger_only_correct=wins,checkout_only_correct=losses,exact_mcnemar_p=p),
        mean_incident_seconds=float(np.mean([r['seconds'] for r in rs])),max_incident_seconds=max(r['seconds'] for r in rs),
        extra_probe_seconds={t:float(np.mean([sum(f['seconds'] for f in r['extra'][t]) for r in rs])) for t in ('checkout','ledger')},
        model_trained=False,scope='Acquisition pilot; no Dreamer performance claim.')
    (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (O/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),methods={n:p.tolist() for n,p in preds.items()}),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':{'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
