"""Development-only nested readout selection; no cluster or evaluation-data access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from microservice.efficiency import decisions
from microservice.mechanisms import FAULT_MASK

CANDIDATES=[dict(kind='linear',penalty=float(p)) for p in (1000,100,10,1)]+[
    dict(kind='rbf',gamma=g,penalty=p) for g in (.1,1.) for p in (.01,.1)]


def kernel(a,b,gamma):
    d=np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1]
    return np.exp(-gamma*d)


def fit(x,y,config):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    gram=z@z.T if config['kind']=='linear' else kernel(z,z,config['gamma'])
    dual=np.linalg.solve(gram+config['penalty']*np.eye(len(x)),y-bias)
    return dict(config=config,mean=mean,scale=scale,bias=bias,z=z,dual=dual)


def score(model,x):
    z=(np.asarray(x)-model['mean'])/model['scale'];c=model['config']
    k=z@model['z'].T if c['kind']=='linear' else kernel(z,model['z'],c['gamma'])
    return k@model['dual']+model['bias']


def diagnose(values):return np.sum((values>0)*np.array([1,2,4]),axis=-1)


def folds(families,n,seed):
    rng=np.random.default_rng(seed);result=[[] for _ in range(n)]
    for family in sorted(set(families)):
        indices=np.where(families==family)[0];rng.shuffle(indices)
        assert len(indices)%n==0
        for k,chunk in enumerate(np.split(indices,n)):result[k].extend(chunk.tolist())
    return [np.array(sorted(chunk),dtype=int) for chunk in result]


def assess(pred,checks,target,families):
    groups={str(f):dict(n=int(sum(families==f)),correct=int(sum((pred==target)&(families==f)))) for f in range(8)}
    return dict(n=len(target),correct=int(sum(pred==target)),mean_checks=float(np.mean(checks)),families=groups,
                gate_passed=all(g['n']>0 and g['correct']>=(g['n'] if f=='0' else math.ceil(.9*g['n'])) for f,g in groups.items()))


def calibrate(values,target,families):
    full=assess(*decisions(values,None),target,families);candidates=[]
    for threshold in [0.,.25,.5,1.,2.,4.,None]:
        stats=assess(*decisions(values,threshold),target,families)
        candidates.append(dict(threshold=threshold,**stats,eligible=stats['gate_passed'] and stats['correct']>=full['correct']-1))
    eligible=[c for c in candidates if c['eligible']]
    chosen=min(eligible,key=lambda c:(c['mean_checks'],-(float('inf') if c['threshold'] is None else c['threshold']))) if eligible else None
    return dict(threshold=chosen['threshold'] if chosen else None,calibration_passed=bool(chosen),candidates=candidates)


def outer_fit(features,y,families,train,test,inner):
    """Only train labels are consulted; test indices are used solely for prediction."""
    n=len(train);target=diagnose(y[train]);selected=[];validation=np.empty((n,4,3));outer=np.empty((len(test),4,3));fixed=np.empty_like(outer);audit=[]
    for t,x in enumerate(features):
        trials=[]
        for config in CANDIDATES:
            cv=np.empty((n,3))
            for held in inner:
                fit_indices=np.setdiff1d(np.arange(n),held)
                model=fit(x[train[fit_indices]],y[train[fit_indices]],config)
                cv[held]=score(model,x[train[held]])
            trials.append(cv)
        correct=[int(sum(diagnose(cv)==target)) for cv in trials]
        best=int(np.argmax(correct));config=CANDIDATES[best]
        validation[:,t]=trials[best];selected.append(config)
        outer[:,t]=score(fit(x[train],y[train],config),x[test])
        fixed[:,t]=score(fit(x[train],y[train],dict(kind='linear',penalty=1.)),x[test])
        audit.append(dict(prefix=t,candidate_correct=correct,selected=best))
    policy=calibrate(validation,target,families[train])
    return dict(scores=outer,fixed_scores=fixed,selected=selected,policy=policy,selection=audit,inner_scores=validation)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'reports/evidence/lab-efficiency-v1/evaluation'
    names=['development.jsonl','development-features.npz','development-scores.json']
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    rows=[json.loads(l) for l in (source/names[0]).read_text().splitlines()];assert len(rows)==80 and all(r['seed']==8201 for r in rows)
    families=np.array([r['fault'] for r in rows]);target=np.array([FAULT_MASK[f] for f in families]);y=((target[:,None]>>np.arange(3))&1)*2-1
    fs=np.load(source/names[1]);original=json.loads((source/names[2]).read_text());outer_splits=folds(families,5,8301)
    result={};predictions={};split_audit=[]
    for k,test in enumerate(outer_splits):
        train=np.setdiff1d(np.arange(80),test);inner=folds(families[train],4,8302+k)
        split_audit.append(dict(train=train.tolist(),test=test.tolist(),inner=[dict(train=train[np.setdiff1d(np.arange(len(train)),held)].tolist(),test=train[held].tolist()) for held in inner]))
    (out/'splits.json').write_text(json.dumps(split_audit,indent=2))
    for name in ['latent','raw']:
        features=[fs[f'{name}_{t}'] for t in range(4)];values=np.empty((80,4,3));fixed=np.empty_like(values);pred=np.zeros(80,int);checks=np.zeros(80,int);details=[]
        for k,test in enumerate(outer_splits):
            train=np.array(split_audit[k]['train']);inner=folds(families[train],4,8302+k)
            run=outer_fit(features,y,families,train,test,inner)
            values[test]=run['scores'];fixed[test]=run['fixed_scores'];pred[test],checks[test]=decisions(run['scores'],run['policy']['threshold'])
            details.append({key:run[key] for key in ['selected','policy','selection']})
            print('OUTER_COMPLETE',name,k,flush=True)
        methods={}
        for label,v in [('tuned',values),('fixed_ridge',fixed),('original_frozen',np.asarray(original[name]))]:
            methods[label]=dict(fixed_budgets={str(t):assess(diagnose(v[:,t]),np.full(80,t),target,families) for t in range(4)})
        methods['tuned']['stopping']=assess(pred,checks,target,families)
        methods['tuned']['calibration_failures']=sum(not d['policy']['calibration_passed'] for d in details)
        v=np.asarray(original[name]);audit={}
        for t in range(4):
            correct=diagnose(v[:,t])==target;margin=np.min(np.abs(v[:,t]),axis=1)
            audit[str(t)]=dict(bit_errors=np.sum((v[:,t]>0)!=(y>0),axis=0).tolist(),wrong_with_margin_at_least_one=int(sum(~correct&(margin>=1))),wrong_margin_mean=float(margin[~correct].mean()) if (~correct).any() else None)
        methods['original_frozen']['error_audit']=audit;result[name]=methods
        predictions[name]=dict(scores=values.tolist(),fixed_scores=fixed.tolist(),stopping_predictions=pred.tolist(),stopping_checks=checks.tolist(),folds=details)
    l=result['latent']['tuned'];r=result['raw']['tuned'];s=l['stopping']
    gate=bool(s['gate_passed'] and s['correct']>=l['fixed_budgets']['3']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    result['development_gate_passed']=gate;result['live_confirmation_warranted']=gate
    result['scope']='Development nested cross-validation only; fixed frozen world model, supervised readout comparison.'
    for name,h in hashes.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
    (out/'inputs.json').write_text(json.dumps(dict(files=hashes,scenario_seed=8201,outer_seed=8301,inner_seed_base=8302,candidates=CANDIDATES),indent=2))
    (out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),families=families.tolist(),methods=predictions)))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('DEVELOPMENT_COMPLETE',gate,flush=True)

if __name__=='__main__':main()
