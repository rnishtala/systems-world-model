import copy
import numpy as np
import pytest
from microservice import live
from microservice.temporal_calibration import FIELDS,GAUGES,DELTAS,FEATURES
from microservice.action_outcomes import schedule,ACTIONS,observations,outcome,prepare,predictions,evaluate

def test_profiles_never_cross_splits_and_actions_are_balanced():
    cases=schedule(123)
    assert cases==schedule(123) and len(cases)==36
    train={c['group'] for c in cases if c['split']=='train'};test={c['group'] for c in cases if c['split']=='test'}
    assert not train&test and len(train)==6 and len(test)==3
    for group in train|test:
        rows=[c for c in cases if c['group']==group]
        assert set(c['action'] for c in rows)==set(ACTIONS)
        assert all(c['profile']==rows[0]['profile'] for c in rows)

def window(t,n,failed=0):
    m={k:1. for k in FIELDS};m['counters']={k:2 for k in GAUGES}
    m['counters'].update({k:n for k in DELTAS});m['counters'].update(completed_requests=n,failed_requests=failed)
    return dict(elapsed=t,metrics={s:copy.deepcopy(m) for s in live.SERVICES})

def test_pre_action_inputs_ignore_future_telemetry():
    windows=[window(3+i*.5,i+1) for i in range(6)]
    before=observations(windows)
    windows.extend([window(6,1000),window(13,9999,9999)])
    assert observations(windows)==before
    assert np.asarray(before).shape==(4,len(FEATURES))

def test_fast_errors_do_not_count_as_successful_recovery():
    ws=[window(12+i*.5,i*10,i*10) for i in range(5)]
    for w in ws:w['metrics']['checkout']['latency_p95']=.001
    values=outcome(ws)
    assert values[0]==1 and values[1]==0 and values[2]==.001

def training_rows():
    rng=np.random.default_rng(4);rows=[]
    for group in range(6):
        for action in ACTIONS:
            rows.append(dict(split='train',group=str(group),action=action,observations=rng.uniform(0,2,(4,len(FEATURES))).tolist(),outcome=[int(action=='none'),float(group+1),.1,2.]))
    return rows

def test_baselines_freeze_and_reject_test_data_for_fitting():
    rows=training_rows();frozen=prepare(rows);saved=copy.deepcopy(frozen)
    test=copy.deepcopy(rows[:4])
    for r in test:r['split']='test';r['predictions']=predictions(frozen,r)
    result=evaluate(test,frozen)
    assert frozen==saved and result['test_trials']==4
    assert set(result['methods'])=={'action_mean','snapshot_ridge','history_ridge'}
    assert all(np.isfinite(list(v['mae'].values())).all() for v in result['methods'].values())
    with pytest.raises(ValueError):prepare(test)
