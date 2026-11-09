import numpy as np
from experiment.exploration import InspectionCoverage


def fake_policy(carry,obs,mode):
    return ({},{},{},{'action':[np.int32(6)]}),{'action':np.array([6],np.int32)},{}


def observation(first=False,last=False):
    return {'is_first':np.array([first]),'is_last':np.array([last])}


def test_guidance_covers_sensors_without_a_fault_label_and_syncs_carry():
    guided=InspectionCoverage(fake_policy,42,1.)
    actions=[];carry=None
    for t in range(3):
        carry,acts,_=guided(carry,observation(first=t==0))
        actions.append(int(acts['action'][0]))
        assert int(carry[-1]['action'][0])==actions[-1]
    assert set(actions[:2])=={1,2}
    assert actions[2] in (3,4,5)


def test_evaluation_never_uses_guidance():
    guided=InspectionCoverage(fake_policy,42,1.)
    carry,acts,_=guided(None,observation(True),mode='eval')
    assert acts['action'][0]==6
    assert guided.stats['overridden_actions']==0


def test_previous_action_uses_explicit_device_transfer():
    import jax
    import jax.numpy as jnp
    with jax.transfer_guard('disallow'):
        scalar=jax.device_put(np.int32(6))
        def device_policy(carry,obs,mode):
            return ({},{},{},{'action':[scalar]}),{'action':np.array([6],np.int32)},{}
        guided=InspectionCoverage(device_policy,42,1.)
        carry,acts,_=guided(None,observation(True))
        stacked=jax.jit(lambda xs:jnp.stack(xs))(carry[-1]['action'])
        np.testing.assert_array_equal(jax.device_get(stacked),acts['action'])
