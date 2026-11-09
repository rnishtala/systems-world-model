"""Deterministic action selection from Dreamer's native actor; no action rules."""
import numpy as np


class GreedyActor:
    def __init__(self,agent):
        import jax
        import ninjax as nj
        self.agent=agent
        @nj.pure
        def select(dyn):
            return agent.model.pol(agent.model.feat2tensor(dyn),1)['action'].pred()
        self.select=jax.jit(lambda params,dyn,key:select(params,dyn,seed=key,create=False,modify=False)[1])

    def init_policy(self,n):return self.agent.init_policy(n)

    def policy(self,carry,obs,mode='eval'):
        import jax
        if mode!='eval':raise ValueError('GreedyActor is evaluation-only')
        carry,_,outs=self.agent.policy(carry,obs,mode='eval')
        with jax.transfer_guard('allow'):
            action=self.select(self.agent.params,self.agent._stack(carry)[1],jax.random.PRNGKey(0))
            carry=(*carry[:-1],{**carry[-1],'action':list(action)})
            actions={'action':np.asarray(action)}
        return carry,actions,outs
