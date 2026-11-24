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


def test_recovery_preserves_frozen_models_and_saved_predictions(tmp_path):
    import json
    from microservice.action_outcomes import restore_completed
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir();cases=schedule(321);rows=[]
    (source/'protocol.json').write_text(json.dumps({'cases':cases}))
    for i,case in enumerate(cases[:25]):
        windows=[window(3+j*.5,j+1) for j in range(6)]+[window(12+j*.5,20+j*2) for j in range(5)]
        result=dict(mechanism=case['mechanism'],relief=case['action'],windows=windows,restored=True)
        (source/f'incident-{i}.json').write_text(json.dumps(result))
        row={k:case[k] for k in ('split','group','action')};row.update(observations=observations(windows),outcome=outcome(windows),restored=True)
        if i==24:
            saved={k:row[k] for k in ('split','group','action','observations')};saved['predictions']=predictions(frozen,row)
            (source/f'prediction-{i}.json').write_text(json.dumps(saved))
        else:rows.append(row)
        if i==23:
            frozen=prepare(rows);(source/'frozen-baselines.json').write_text(json.dumps(frozen))
    restored,model=restore_completed(source,out,cases)
    assert len(restored)==25 and model==frozen
    assert (out/'frozen-baselines.json').read_bytes()==(source/'frozen-baselines.json').read_bytes()
    assert (out/'prediction-24.json').read_bytes()==(source/'prediction-24.json').read_bytes()
    saved['observations'][0][0]+=1;(source/'prediction-24.json').write_text(json.dumps(saved))
    other=tmp_path/'bad';other.mkdir()
    with pytest.raises(ValueError,match='inputs mismatch'):restore_completed(source,other,cases)


def test_fresh_validation_uses_frozen_models_without_training(tmp_path,monkeypatch):
    import json,hashlib,sys
    import microservice.action_outcomes as m
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out'
    data=json.dumps(prepare(training_rows())).encode()
    (source/'frozen-baselines.json').write_bytes(data)
    (source/'summary.json').write_text(json.dumps({'action_outcomes':{'all_restored':True,'frozen_sha256':hashlib.sha256(data).hexdigest()}}))
    (source/'protocol.json').write_text(json.dumps({'seed':123,'cases':schedule(123)}))
    cases=m.validation_schedule(456,schedule(123))
    assert len(cases)==36 and len({c['group'] for c in cases})==9
    assert all(c['split']=='test' for c in cases)
    for group in {c['group'] for c in cases}:
        assert {c['action'] for c in cases if c['group']==group}==set(ACTIONS)
    def fake_incident(mechanism,action,profile,before_action):
        windows=[window(3+j*.5,j+1) for j in range(6)]
        before_action(windows)
        assert len(list(out.glob('prediction-*.json')))==len(list(out.glob('incident-*.json')))+1
        windows.extend(window(12+j*.5,20+j*2) for j in range(5))
        return dict(mechanism=mechanism,relief=action,windows=windows,restored=True)
    def forbidden(*args):raise AssertionError('Validation must not refit')
    monkeypatch.setattr(m,'incident',fake_incident);monkeypatch.setattr(m,'prepare',forbidden)
    monkeypatch.setenv('POD_NAMESPACE','systems-world-model-lab')
    monkeypatch.setattr(sys,'argv',['action_outcomes','--out',str(out),'--seed','456','--validate-from',str(source)])
    m.main()
    result=json.loads((out/'summary.json').read_text())['action_outcomes']
    assert result['training_trials']==0 and result['test_trials']==36 and result['independent_test_profiles']==9
    assert (out/'frozen-baselines.json').read_bytes()==data
    with pytest.raises(ValueError,match='overlaps'):m.validation_schedule(456,cases)
    bad=tmp_path/'bad';bad.mkdir()
    (source/'frozen-baselines.json').write_bytes(data+b' ')
    with pytest.raises(ValueError,match='Invalid frozen'):m.load_validation(source,bad,456)
