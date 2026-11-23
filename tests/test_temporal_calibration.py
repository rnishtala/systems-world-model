import copy
import numpy as np
from microservice.temporal_calibration import schedule,fit_tree,predict,evaluate,snapshot_features,FIELDS,GAUGES,DELTAS,FEATURES
from microservice import live

def test_separate_incidents_and_deterministic_schedule():
    cases=schedule(42)
    assert cases==schedule(42) and cases!=schedule(43)
    assert len(cases)==36
    for split in ('train','test'):
        for mechanism in ('healthy','database','upstream'):
            assert sum(c['split']==split and c['mechanism']==mechanism for c in cases)==6
    assert all(c['profile']['duration']<45 and c['profile']['relief_at']>c['profile']['duration'] for c in cases)

def test_tree_fits_three_classes_without_mutating_on_test():
    x=np.array([[i] for i in range(12)]);y=[0]*4+[1]*4+[2]*4
    tree=fit_tree(x,y);frozen=copy.deepcopy(tree)
    assert [predict(tree,v) for v in x]==y
    result=evaluate(tree,[dict(label=l,features=v) for l,v in zip(y,x)])
    assert result['correct']==12 and result['confusion']==[[4,0,0],[0,4,0],[0,0,4]]
    assert tree==frozen and 0<result['wilson95'][0]<1

def test_snapshot_excludes_controls_and_cumulative_history():
    m={k:1. for k in FIELDS};m['counters']={k:20 for k in GAUGES+DELTAS}
    old=copy.deepcopy(m)
    for k in DELTAS:old['counters'][k]-=2
    windows=[dict(elapsed=9.5,metrics={s:copy.deepcopy(old) for s in live.SERVICES}),dict(elapsed=10,metrics={s:copy.deepcopy(m) for s in live.SERVICES})]
    first=snapshot_features(windows)
    assert len(first)==len(FEATURES)
    for w in windows:
        w['injected_label']='database'
        for s in live.SERVICES:
            for k in DELTAS:w['metrics'][s]['counters'][k]+=10000
    assert snapshot_features(windows)==first
