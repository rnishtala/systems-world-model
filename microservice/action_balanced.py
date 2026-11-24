"""Stratified workload coverage and separate repeatability diagnostics."""
import random,statistics
from microservice.action_outcomes import ACTIONS,TARGETS,evaluate

BANDS={'below':(.080,.090),'near':(.097,.103),'above':(.110,.120)}

def schedule(seed):
    cases=[]
    for split,n,offset in [('train',2,0),('test',1,1)]:
        rng=random.Random(seed+offset);part=[]
        for band,(low,high) in BANDS.items():
            for load in ('low','high'):
                for j in range(n):
                    hold=rng.uniform(low,high)
                    # Offered load relative to nominal two-slot database capacity.
                    rate=(2/hold)*rng.uniform(*((.35,.55) if load=='low' else (1.3,1.6)))
                    profile=dict(healthy_hold=rng.uniform(.03,.05),fault_hold=hold,base_rate=rate,burst_rate=rng.uniform(65,115),onset=rng.uniform(2,3),duration=16,relief_at=6)
                    for action in ACTIONS:part.append(dict(split=split,group=f'{split}-database-{band}-{load}-{j}',mechanism='database',action=action,regime=band+'-'+load,profile=profile.copy()))
        for mechanism in ('healthy','upstream'):
            for j in range(2):
                profile=dict(healthy_hold=rng.uniform(.03,.05),fault_hold=rng.uniform(.08,.12),base_rate=rng.uniform(18,34),burst_rate=rng.uniform(65,115),onset=rng.uniform(2,3),duration=16,relief_at=6)
                for action in ACTIONS:part.append(dict(split=split,group=f'{split}-{mechanism}-{j}',mechanism=mechanism,action=action,regime=mechanism,profile=profile.copy()))
        rng.shuffle(part);cases.extend(part)
    repeats=[]
    for band in ('below','above'):
        original=next(c for c in cases if c['group']==f'train-database-{band}-high-0')
        for repeat in range(2):
            for action in ACTIONS:
                repeats.append(dict(split='diagnostic',group='repeat-'+band,mechanism='database',action=action,regime=band+'-high',repeat=repeat,profile=original['profile'].copy()))
    random.Random(seed+2).shuffle(repeats)
    return cases+repeats

def report(rows,cases,frozen):
    test=[dict(r,regime=c['regime']) for r,c in zip(rows,cases) if r['split']=='test']
    repeats=[r for r in rows if r['split']=='diagnostic'];pairs=[]
    for group in sorted({r['group'] for r in repeats}):
        for action in ACTIONS:
            pair=[r for r in repeats if r['group']==group and r['action']==action]
            if len(pair)!=2:raise ValueError('Expected two diagnostic repeats')
            pairs.append(dict(group=group,action=action,outcome_absolute_difference={target:abs(pair[0]['outcome'][j]-pair[1]['outcome'][j]) for j,target in enumerate(TARGETS)}))
    diffs=[]
    for group in sorted({r['group'] for r in test}):
        rs=[r for r in test if r['group']==group]
        diffs.append(dict(group=group,history_minus_snapshot=statistics.mean(abs(r['predictions']['history_ridge'][0]-r['outcome'][0])-abs(r['predictions']['snapshot_ridge'][0]-r['outcome'][0]) for r in rs)))
    return dict(by_regime={regime:evaluate([r for r in test if r['regime']==regime],frozen) for regime in sorted({r['regime'] for r in test})},paired_profiles=diffs,diagnostic_trials=len(repeats),repeat_differences=pairs)
