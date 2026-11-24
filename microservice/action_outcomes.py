"""Bounded action-outcome screening: grouped splits and predictions before intervention."""
import argparse,json,os,random,hashlib,shutil
from pathlib import Path
import numpy as np
from microservice import live
from microservice.temporal_pilot import incident
from microservice.temporal_calibration import FIELDS,GAUGES,DELTAS,FEATURES

ACTIONS=('none','database','load','retry')
TARGETS=('error_fraction','successful_requests_per_second','latency_seconds','ledger_queue')
PENALTIES=(1.,10.,100.)

def schedule(seed):
    cases=[]
    for split,n,offset in [('train',2,0),('test',1,1)]:
        rng=random.Random(seed+offset);part=[]
        for mechanism in ('healthy','database','upstream'):
            for j in range(n):
                group=f'{split}-{mechanism}-{j}'
                profile=dict(healthy_hold=rng.uniform(.03,.05),fault_hold=rng.uniform(.08,.12),base_rate=rng.uniform(18,34),burst_rate=rng.uniform(65,115),onset=rng.uniform(2,3),duration=16,relief_at=6)
                for action in ACTIONS:part.append(dict(split=split,group=group,mechanism=mechanism,action=action,profile=profile.copy()))
        rng.shuffle(part);cases.extend(part)
    return cases

def observations(windows):
    # Actual action is invoked only after this prefix is captured. No controls or labels.
    prefix=[w for w in windows if w['elapsed']<6]
    if len(prefix)<5:raise ValueError('Insufficient pre-action history')
    rows=[]
    for previous,now in zip(prefix[-5:-1],prefix[-4:]):
        dt=now['elapsed']-previous['elapsed']
        if not .1<=dt<=2:raise ValueError('Invalid telemetry interval')
        values=[]
        for s in live.SERVICES:
            m=now['metrics'][s];c=m['counters'];old=previous['metrics'][s]['counters']
            values.extend(m[k] for k in FIELDS)
            values.extend(c.get(k,0) for k in GAUGES)
            # Rates remove collection-window jitter. Window count remains an observation.
            values.extend((c.get(k,0)-old.get(k,0))/dt for k in DELTAS)
        rows.append(values)
    rows=np.asarray(rows,float)
    if not np.isfinite(rows).all() or np.any(rows<0):raise ValueError('Invalid telemetry')
    return rows.tolist()

def outcome(windows,low=12,high=16):
    ws=[w for w in windows if low<=w['elapsed']<high]
    if len(ws)<3:raise ValueError('Insufficient outcome windows')
    first,last=ws[0],ws[-1];dt=last['elapsed']-first['elapsed']
    a=first['metrics']['checkout']['counters'];b=last['metrics']['checkout']['counters']
    n=b.get('completed_requests',0)-a.get('completed_requests',0)
    failed=b.get('failed_requests',0)-a.get('failed_requests',0)
    if n<=0 or not 0<=failed<=n:raise ValueError('No valid outcome completions')
    return [failed/n,(n-failed)/dt,float(np.mean([w['metrics']['checkout']['latency_p95'] for w in ws])),float(np.mean([w['metrics']['ledger']['counters'].get('pool_waiters',0) for w in ws]))]

def design(row,history=False):
    h=np.asarray(row['observations']);base=np.log1p(h.ravel() if history else h[-1])
    a=np.eye(4)[ACTIONS.index(row['action'])]
    return np.r_[a,np.outer(a,base).ravel()]

def fit_ridge(x,y,penalty):
    x=np.asarray(x);y=np.asarray(y);mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    # Dual solve keeps grouped CV small even with history/action interactions.
    weights=z.T@np.linalg.solve(z@z.T+penalty*np.eye(len(z)),y-bias)
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=weights.tolist(),bias=bias.tolist())

def ridge_predict(model,x):
    return ((np.asarray(x)-model['mean'])/model['scale'])@np.asarray(model['weights'])+model['bias']

def clip(y):
    y=np.maximum(y,0);y[...,0]=np.minimum(y[...,0],1);return y

def prepare(rows):
    if any(r['split']!='train' for r in rows):raise ValueError('Training data only')
    y=np.array([r['outcome'] for r in rows]);groups=np.array([r['group'] for r in rows]);models={};cv={}
    models['action_mean']={a:y[[r['action']==a for r in rows]].mean(0).tolist() for a in ACTIONS}
    # Select regularization independently using whole-profile held-out training folds.
    for name,history in [('snapshot_ridge',False),('history_ridge',True)]:
        x=np.array([design(r,history) for r in rows]);scores=[]
        for penalty in PENALTIES:
            pred=np.zeros_like(y)
            for group in np.unique(groups):
                held=groups==group
                pred[held]=clip(ridge_predict(fit_ridge(x[~held],y[~held],penalty),x[held]))
            scores.append(float(np.abs(pred[:,0]-y[:,0]).mean()))
        best=int(np.argmin(scores));models[name]=fit_ridge(x,y,PENALTIES[best]);cv[name]=dict(penalty=PENALTIES[best],error_fraction_mae=scores[best])
    pred=[]
    for row in rows:
        other=[r['outcome'] for r in rows if r['group']!=row['group'] and r['action']==row['action']]
        pred.append(np.mean(other,axis=0))
    cv['action_mean']=dict(error_fraction_mae=float(np.abs(np.asarray(pred)[:,0]-y[:,0]).mean()))
    return dict(models=models,cv=cv,selected=min(cv,key=lambda k:cv[k]['error_fraction_mae']))

def predictions(frozen,row):
    return dict(action_mean=frozen['models']['action_mean'][row['action']],**{name:clip(ridge_predict(frozen['models'][name],design(row,history))).tolist() for name,history in [('snapshot_ridge',False),('history_ridge',True)]})

def evaluate(rows,frozen):
    y=np.asarray([r['outcome'] for r in rows]);metrics={}
    for name in frozen['models']:
        pred=np.asarray([r['predictions'][name] for r in rows]);errors=np.abs(pred-y)
        metrics[name]=dict(mae=dict(zip(TARGETS,errors.mean(0).tolist())),by_action={a:dict(zip(TARGETS,errors[[r['action']==a for r in rows]].mean(0).tolist())) for a in ACTIONS})
    return dict(test_trials=len(rows),independent_test_profiles=len({r['group'] for r in rows}),selected=frozen['selected'],cv=frozen['cv'],methods=metrics)

def restore_completed(source,out,cases):
    """Resume only after training froze; preserve completed trials and original predictions."""
    source=Path(source);out=Path(out)
    original=json.loads((source/'protocol.json').read_text())
    if original['cases']!=cases or (source/'summary.json').exists():raise ValueError('Recovery source protocol mismatch or already complete')
    present=sorted(int(p.stem.split('-')[1]) for p in source.glob('incident-*.json'))
    if not 24<=len(present)<36 or present!=list(range(len(present))):raise ValueError('Recovery requires a contiguous completed training prefix')
    frozen=json.loads((source/'frozen-baselines.json').read_text());rows=[]
    for i in present:
        case=cases[i];result=json.loads((source/f'incident-{i}.json').read_text())
        if not result['restored'] or result['mechanism']!=case['mechanism'] or result['relief']!=case['action']:raise ValueError('Invalid restored trial')
        row={k:case[k] for k in ('split','group','action')};row['observations']=observations(result['windows'])
        if case['split']=='test':
            saved=json.loads((source/f'prediction-{i}.json').read_text())
            if any(saved[k]!=row[k] for k in row):raise ValueError('Saved prediction inputs mismatch')
            expected=predictions(frozen,row)
            if set(saved['predictions'])!=set(expected) or any(not np.allclose(saved['predictions'][k],expected[k],rtol=1e-10,atol=1e-10) for k in expected):raise ValueError('Saved prediction model mismatch')
            row['predictions']=saved['predictions']
            shutil.copyfile(source/f'prediction-{i}.json',out/f'prediction-{i}.json')
        row.update(outcome=outcome(result['windows']),restored=True);rows.append(row)
        shutil.copyfile(source/f'incident-{i}.json',out/f'incident-{i}.json')
    shutil.copyfile(source/'frozen-baselines.json',out/'frozen-baselines.json')
    return rows,frozen

def validation_schedule(seed,source_cases):
    cases=[]
    for batch in range(3):
        for case in schedule(seed+batch*1009)[24:]:
            case['group']=f"validation-{batch}-"+case['group']
            cases.append(case)
    known={json.dumps(c['profile'],sort_keys=True) for c in source_cases}
    if any(json.dumps(c['profile'],sort_keys=True) in known for c in cases):raise ValueError('Validation profile overlaps source')
    return cases

def load_validation(source,out,seed):
    source=Path(source);out=Path(out)
    summary=json.loads((source/'summary.json').read_text())['action_outcomes']
    protocol=json.loads((source/'protocol.json').read_text())
    data=(source/'frozen-baselines.json').read_bytes()
    if not summary['all_restored'] or hashlib.sha256(data).hexdigest()!=summary['frozen_sha256']:raise ValueError('Invalid frozen validation source')
    if seed==protocol['seed']:raise ValueError('Use a fresh validation seed')
    cases=validation_schedule(seed,protocol['cases'])
    (out/'frozen-baselines.json').write_bytes(data)
    return cases,json.loads(data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--seed',required=True,type=int);sources=p.add_mutually_exclusive_group();sources.add_argument('--resume-from');sources.add_argument('--validate-from');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cases=schedule(a.seed)
    protocol=dict(seed=a.seed,scope='Action-outcome feasibility screening; no Dreamer training or causal policy claim',cases=cases,features=FEATURES,actions=ACTIONS,targets=TARGETS,action_seconds=6,outcome_seconds=[12,16],primary='Mean absolute error in post-action request error fraction',penalties=PENALTIES,selection='Leave-profile-out training CV; freeze before all test trials',limits='24 training trials from 6 profiles; 12 test trials from only 3 profiles. Actions recreate profiles, not identical initial states.')
    rows=[];frozen=None
    if a.validate_from:
        cases,frozen=load_validation(a.validate_from,out,a.seed)
        protocol.update(cases=cases,validation=dict(source_run=Path(a.validate_from).parent.name,new_profiles=9,training_trials=0),limits='36 test trials from 9 independent profiles; unchanged frozen predictors; same workload ranges as screening.')
    if a.resume_from:
        rows,frozen=restore_completed(a.resume_from,out,cases)
        protocol['recovery']=dict(source_run=Path(a.resume_from).parent.name,preserved_trials=len(rows),frozen_sha256=hashlib.sha256((out/'frozen-baselines.json').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for i,case in enumerate(cases):
        if i<len(rows):continue
        row={k:case[k] for k in ('split','group','action')}
        def before_action(windows):
            row['observations']=observations(windows)
            if case['split']=='test':
                row['predictions']=predictions(frozen,row)
                (out/f'prediction-{i}.json').write_text(json.dumps(row))
        result=incident(case['mechanism'],case['action'],case['profile'],before_action=before_action)
        (out/f'incident-{i}.json').write_text(json.dumps(result))
        row.update(outcome=outcome(result['windows']),restored=result['restored']);rows.append(row)
        if i==23 and not a.validate_from:
            frozen=prepare(rows);(out/'frozen-baselines.json').write_text(json.dumps(frozen))
        print('ACTION_OUTCOMES_PROGRESS',i+1,'/36',flush=True)
    (out/'examples.json').write_text(json.dumps(rows))
    scores=evaluate([r for r in rows if r['split']=='test'],frozen)
    summary=dict(scope=protocol['scope'],incidents=[],action_outcomes=dict(**scores,training_trials=0 if a.validate_from else 24,validation=protocol.get('validation'),recovery=protocol.get('recovery'),all_restored=all(r['restored'] for r in rows),frozen_sha256=hashlib.sha256((out/'frozen-baselines.json').read_bytes()).hexdigest()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('ACTION_OUTCOMES_COMPLETE',flush=True)

if __name__=='__main__':main()
