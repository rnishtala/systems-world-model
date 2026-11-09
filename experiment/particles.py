"""Vote across native Dreamer latent-state particles; no expert action rules."""
import numpy as np


class ParticleActor:
    def __init__(self,agent,particles=16):
        if particles<1:raise ValueError('particles must be positive')
        self.agent=agent;self.particles=particles

    def init_policy(self,n):
        if n!=1:raise ValueError('One external environment is supported')
        return self.agent.init_policy(self.particles)

    def policy(self,carry,obs,mode='eval'):
        import jax
        if mode!='eval':raise ValueError('ParticleActor is evaluation-only')
        expanded={k:np.repeat(v,self.particles,axis=0) for k,v in obs.items()}
        carry,acts,_=self.agent.policy(carry,expanded,mode='eval')
        action=int(np.bincount(acts['action'],minlength=7).argmax())
        previous=[jax.device_put(np.int32(action),x.sharding) for x in carry[-1]['action']]
        carry=(*carry[:-1],{**carry[-1],'action':previous})
        return carry,{'action':np.array([action],np.int32)},{}
