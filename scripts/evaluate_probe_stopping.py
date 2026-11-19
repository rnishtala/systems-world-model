"""Nested training-only stopping screen; no held-out experiment data access."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from microservice.partial_readout import CANDIDATES,fit,score,diagnose
from microservice.latent_probe import labels
from microservice.probe_target import frames,vector
R=Path(__file__).resolve().parents[1]
O=R/'reports/evidence/probe-stopping-v1'
NAMES=('latent','history','summary_history')
THRESHOLDS=(0.,.25,.5,1.,2.,4.,None)

def decide(values,threshold):
    values=np.asarray(values);n,t,_=values.shape
    stop=np.full(n,t-1,dtype=int)
    if threshold is not None:
        eligible=np.min(np.abs(values),axis=-1)>=threshold
        stop=np.where(eligible.any(1),eligible.argmax(1),t-1)
    return diagnose(values[np.arange(n),stop]),stop

def metrics(pred,stop,y,families,cost):
    target=diagnose(y)
    groups={str(f):dict(n=int(sum(families==f)),correct=int(sum((pred==target)&(families==f)))) for f in range(8)}
    reliable=(sum(pred==target)>=math.ceil(.9*len(y)) and groups['0']['correct']==groups['0']['n'] and
              all(g['n']>0 and g['correct']>=math.ceil(.75*g['n']) for g in groups.values()))
    return dict(n=len(y),correct=int(sum(pred==target)),mean_windows=float(np.mean(stop+1)),
                mean_requests=float(np.mean(cost[np.arange(len(y)),stop])),families=groups,reliable=bool(reliable))

def calibrate(values,y,families,cost):
    full=metrics(*decide(values,None),y,families,cost);trials=[]
    for threshold in THRESHOLDS:
        m=metrics(*decide(values,threshold),y,families,cost)
        trials.append(dict(threshold=threshold,**m,eligible=m['reliable'] and m['correct']>=full['correct']-1))
    eligible=[c for c in trials if c['eligible']]
    best=min(eligible,key=lambda c:(c['mean_windows'],-(float('inf') if c['threshold'] is None else c['threshold']))) if eligible else None
    return dict(threshold=best['threshold'] if best else None,calibration_passed=bool(best),candidates=trials)

def outer(features,y,groups,families,cost,train,held):
    validation=np.empty((len(train),8,3));prediction=np.empty((len(held),8,3));selection=[]
    for t,x in enumerate(features):
        trials=[]
        for config in CANDIDATES:
            v=np.empty((len(train),3))
            for group in sorted(set(groups[train])):
                test=groups[train]==group
                v[test]=score(fit(x[train[~test]],y[train[~test]],config),x[train[test]])
            trials.append(v)
        counts=[int(sum(diagnose(v)==diagnose(y[train]))) for v in trials]
        best=int(np.argmax(counts));config=CANDIDATES[best]
        validation[:,t]=trials[best]
        prediction[:,t]=score(fit(x[train],y[train],config),x[held])
        selection.append(dict(prefix=t+1,config=config,inner_correct=counts))
    policy=calibrate(validation,y[train],families[train],cost[train])
    return prediction,dict(selection=selection,policy=policy)

def main():
    rows=[json.loads(l) for l in (O/'collection/train.jsonl').read_text().splitlines()]
    assert len(rows)==64 and {r['seed'] for r in rows}=={9101,9103}
    y=labels(rows);groups=np.array([r['repeat'] for r in rows]);families=np.array([r['fault'] for r in rows])
    cost=np.array([np.cumsum([f['requests'] for f in frames(r)]) for r in rows])
    fs=np.load(O/'evaluation/prefix-features.npz',allow_pickle=False)
    for t in range(8):
        expected=np.array([np.concatenate([vector(r,j) for j in range(t+1)]) for r in rows])
        np.testing.assert_array_equal(fs[f'history_{t}'],expected)
    folds=[dict(group=int(g),train=np.where(groups!=g)[0].tolist(),held=np.where(groups==g)[0].tolist()) for g in sorted(set(groups))]
    results={};predictions={}
    for name in NAMES:
        features=[fs[f'{name}_{t}'] for t in range(8)]
        values=np.empty((64,8,3));pred=np.empty(64,int);stops=np.empty(64,int);details=[]
        for fold in folds:
            train=np.array(fold['train']);held=np.array(fold['held'])
            values[held],audit=outer(features,y,groups,families,cost,train,held)
            pred[held],stops[held]=decide(values[held],audit['policy']['threshold']);details.append(audit)
        results[name]=dict(stopping=metrics(pred,stops,y,families,cost),calibration_failures=sum(not d['policy']['calibration_passed'] for d in details),
            fixed_prefixes={str(t+1):metrics(diagnose(values[:,t]),np.full(64,t),y,families,cost) for t in range(8)})
        predictions[name]=dict(scores=values.tolist(),predictions=pred.tolist(),stops=stops.tolist(),folds=details)
        print('STOPPING_RESULT',name,json.dumps(results[name]['stopping']),flush=True)
    l=results['latent']['stopping'];controls=[results[n]['stopping'] for n in NAMES[1:]]
    gate=bool(l['reliable'] and l['correct']>=results['latent']['fixed_prefixes']['8']['correct']-1 and
              all(l['correct']>=c['correct']-1 and c['mean_windows']-l['mean_windows']>=.5 for c in controls) and 8-l['mean_windows']>=1)
    result=dict(scope='Optimistic development screen: world-model weights trained on these 64 incidents; nested readout/stopping selection only.',methods=results,development_gate=gate,fresh_confirmation_warranted=gate,
        cost_scope='Shadow prefix estimates; warmup excluded; fixed schedule, not adaptive action selection.')
    for name,value in [('summary',result),('predictions',dict(target=diagnose(y).tolist(),methods=predictions)),('splits',folds)]:
        (O/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    paths=[O/'evaluation/prefix-features.npz',O/'collection/train.jsonl',R/'reports/PROBE_STOPPING_PROTOCOL.md',Path(__file__)]
    (O/'analysis-input-hashes.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print('DEVELOPMENT_GATE',gate,flush=True)
if __name__=='__main__':main()
