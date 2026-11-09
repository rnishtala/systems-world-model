import numpy as np
from experiment.particles import ParticleActor


def test_particle_votes_share_observations_and_executed_action():
    import jax
    class Agent:
        def init_policy(self,n):return n
        def policy(self,carry,obs,mode):
            assert carry==16 and obs['vector'].shape==(16,9)
            np.testing.assert_array_equal(obs['vector'],np.ones((16,9)))
            sampled=np.array([1]*12+[4]*4,np.int32)
            state=({},{},{},{'action':[jax.device_put(x) for x in sampled]})
            return state,{'action':sampled},{}
    wrapper=ParticleActor(Agent())
    carry,acts,_=wrapper.policy(wrapper.init_policy(1),{'vector':np.ones((1,9))})
    assert acts['action'].tolist()==[1]
    assert all(int(jax.device_get(x))==1 for x in carry[-1]['action'])
