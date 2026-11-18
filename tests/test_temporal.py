import copy
import numpy as np
from microservice.temporal import vector,TemporalIncident


def row():
    return {'fault':7,'frames':[{'vector':[.01*(i+1) for i in range(10)]} for _ in range(4)]}


def test_resource_counters_and_labels_never_enter_view():
    original=row();changed=copy.deepcopy(original);changed['fault']=0
    for f in changed['frames']:
        for i in (3,4,5,6):f['vector'][i]=999
    for t in range(4):np.testing.assert_array_equal(vector(original,t),vector(changed,t))


def test_prefix_does_not_read_future():
    original=row();changed=copy.deepcopy(original)
    changed['frames'][3]['vector']=[999]*10
    for t in range(3):np.testing.assert_array_equal(vector(original,t),vector(changed,t))
    assert not np.array_equal(vector(original,3),vector(changed,3))


def test_wait_sequence_matches_environment_and_supervision_boundary():
    r=row();sim=TemporalIncident(r['frames'],7,compositional=True)
    np.testing.assert_array_equal(sim.observe(),vector(r,0))
    for t in range(1,4):
        observed,reward,done=sim.step(0)
        np.testing.assert_array_equal(observed,vector(r,t))
        assert reward==-.05 and not done
        assert np.all(observed[10:13]==float(t==3))
    _,reward,done=sim.step(11)
    assert done and reward==2.95
