import numpy as np
from experiment.environment import QueueTriage


def test_passive_observations_do_not_leak_fault_identity():
    a,b=QueueTriage(42),QueueTriage(42)
    np.testing.assert_array_equal(a.reset(1),b.reset(2))
    for _ in range(3):
        x,ra,_=a.step(0);y,rb,_=b.step(0)
        np.testing.assert_array_equal(x,y)
        assert ra==rb


def test_checks_reveal_distinguishing_evidence_and_cost():
    a,b=QueueTriage(1),QueueTriage(1)
    a.reset(1);b.reset(2)
    a.step(1);b.step(1)
    assert a.cpu>.65>b.cpu
    assert a.checks==1 and b.checks==1


def test_episodes_are_bounded_and_diagnosis_is_scored():
    a=QueueTriage(0);a.reset(2)
    _,reward,done=a.step(5)
    assert done and reward>0
    a.reset(2)
    for _ in range(6):a.step(0)
    assert a.done and a.choice==-1


def test_abstention_ablation_changes_only_abstention_reward():
    a,b=QueueTriage(42),QueueTriage(42,abstention_penalty=2.)
    np.testing.assert_array_equal(a.reset(1),b.reset(1))
    x,ra,_=a.step(1);y,rb,_=b.step(1)
    np.testing.assert_array_equal(x,y)
    assert ra==rb
    x,ra,_=a.step(6);y,rb,_=b.step(6)
    np.testing.assert_array_equal(x,y)
    assert np.isclose(ra-rb,1.5)
