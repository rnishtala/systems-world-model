import numpy as np
import pytest
from microservice.environment import Incident
from microservice.mechanisms import FAULT_MASK, diagnosis_name

@pytest.mark.parametrize('fault',range(8))
def test_all_mechanisms_reward_exact_set_without_observation_leakage(fault):
    frames=[{'vector':[.1]*10}]*4
    reference=Incident(frames,0,compositional=True)
    candidate=Incident(frames,fault,compositional=True)
    for action in (1,2,3):
        np.testing.assert_array_equal(reference.observe(),candidate.observe())
        assert reference.step(action)[1]==candidate.step(action)[1]
    rewards=[Incident(frames,fault,compositional=True).step(4+mask)[1] for mask in range(8)]
    assert np.argmax(rewards)==FAULT_MASK[fault]
    for mask in range(8):
        assert rewards[mask]==pytest.approx(2.95-2*((mask^FAULT_MASK[fault]).bit_count()))

def test_escalation_and_old_contract_remain_distinct():
    frames=[{'vector':[0]*10}]*4
    old=Incident(frames,3)
    assert old.step(7)[1]>0
    new=Incident(frames,3,compositional=True)
    assert new.step(8)[1]>0 # delay bit=4
    assert Incident(frames,0,compositional=True).step(12)[1]<0
    with pytest.raises(ValueError):Incident(frames,0).step(12)
    assert diagnosis_name(6)=='pool+delay'

def test_triple_control_preserves_all_mechanisms(monkeypatch):
    from microservice import live
    calls=[]
    monkeypatch.setattr(live,'reset',lambda:None)
    monkeypatch.setattr(live,'request',lambda service,path,body,control:calls.append((service,body)))
    live.configure(7,np.random.default_rng(123),True)
    assert any(s=='payment' and d.get('cpu_seconds',0)>0 for s,d in calls)
    ledger=[d for s,d in calls if s=='ledger'][-1]
    assert ledger['pool_size']==1 and ledger['delay_seconds']>0 and ledger['db_hold']>0

def test_live_cli_decodes_versioned_answers():
    from microservice.mechanisms import describe_action
    assert describe_action(8,'single-v1')==(True,'escalation')
    assert describe_action(8,'mechanisms-v1')==(True,'delay')
    assert describe_action(11,'mechanisms-v1')==(True,'cpu+pool+delay')
    assert describe_action(12,'mechanisms-v1')==(True,'escalation')
    with pytest.raises(ValueError):describe_action(12,'single-v1')
    with pytest.raises(ValueError):describe_action(1,'unknown')
