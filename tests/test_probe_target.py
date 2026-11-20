import copy,json,sys,types
from pathlib import Path
import numpy as np
import pytest
from microservice.probe_target import frames,vector,next_action,ProbeEnv

def row():
    from tests.fixtures import incident
    return incident()


def test_no_future_truth_or_counter_leakage():
    a=row();b=copy.deepcopy(a);b['fault']=999;b['seed']=0
    b['base'][1]['metrics']['ledger']['latency_p95']=999
    for f in frames(b):
        for m in f['metrics'].values():m.update(cpu_mean=999,pool_wait_p95=999,db_mean=999)
    np.testing.assert_array_equal(vector(a,0),vector(b,0))
    for t,f in enumerate(frames(a)):
        x=vector(a,t);assert x.shape==(14,) and x.dtype==np.float32
        if f['target']=='ledger':np.testing.assert_array_equal(x[1:4],np.zeros(3))
        assert bool(np.all(x[10:13]))==(t==7)
        if t<7:
            target=frames(a)[t+1];np.testing.assert_allclose(next_action(a,t),[target['target']=='ledger',target['requests']/12])
    np.testing.assert_array_equal(next_action(a,7),[0,0])

def test_only_measured_actions_replay(monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'elements',types.SimpleNamespace(Space=lambda *a,**kw:None))
    a=row();p=tmp_path/'rows.jsonl';p.write_text(json.dumps(a)+'\n')
    env=ProbeEnv(p);env.step(dict(reset=True,action=np.zeros(2,np.float32)))
    for t in range(1,9):
        obs=env.step(dict(reset=False,action=next_action(a,t-1)))
        np.testing.assert_array_equal(obs['vector'],vector(a,t));assert obs['is_last']==(t==8)
    env.step(dict(reset=True,action=np.zeros(2,np.float32)))
    wrong=next_action(a,0).copy();wrong[0]=1-wrong[0]
    with pytest.raises(ValueError,match='counterfactual'):env.step(dict(reset=False,action=wrong))
    with pytest.raises(ValueError):env.step(dict(reset=False,action=np.zeros(1,np.float32)))


def test_partial_supervision_uses_six_and_eight_windows_without_terminal():
    from microservice.supervision import diagnosis_mask, telemetry
    a=row()
    observations=np.stack([vector(a,t) for t in range(9)])
    terminal=np.arange(9)==8
    np.testing.assert_array_equal(diagnosis_mask(observations,terminal),
                                  [False,False,False,False,False,True,False,True,False])
    assert 'mechanism_target' not in telemetry(dict(vector=observations,mechanism_target=np.ones((9,3))))
