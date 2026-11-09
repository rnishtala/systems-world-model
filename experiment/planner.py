"""Frozen Dreamer world-model lookahead with explicit bounded-action rules.

This is a separate model-based controller, not the native learned actor. It
allows each inspection and waiting at most once, then must diagnose or escalate.
Those bounds are harness rules, not learned efficiency. No hidden fault labels
or simulator transitions are queried by planning.
"""
import numpy as np


def combine_values(first, continuation, last, gathering):
    # Average stochastic continuations BEFORE selecting the best diagnosis.
    future=last.mean(axis=-1).max(axis=-1)
    return (first+gathering[:,None]*continuation*future).mean(axis=-1)


def choose_action(values, vector, waited):
    values=np.asarray(values).copy()
    if waited:values[0]=-np.inf
    if vector[6]:values[1]=-np.inf
    if vector[7]:values[2]=-np.inf
    return int(values.argmax())


class WorldModelPlanner:
    def __init__(self, agent, seed=0, particles=16, continuations=4):
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        self.agent=agent;self.seed=seed;self.calls=0
        self.waited=False
        p,q=particles,continuations

        @nj.pure
        def lookahead(dyn):
            starts=jax.tree.map(lambda x:jnp.repeat(x,7*p,axis=0),dyn)
            actions={'action':jnp.repeat(jnp.arange(7,dtype=jnp.int32),p)}
            next_dyn,(feat,_)=agent.model.dyn.imagine(starts,actions,1,False,single=True)
            first=agent.model.rew(agent.model.feat2tensor(feat),1).pred().reshape(7,p)
            cont=agent.model.con(agent.model.feat2tensor(feat),1).prob(1).reshape(7,p)
            branches=jax.tree.map(lambda x:jnp.repeat(x,3*q,axis=0),next_dyn)
            endings={'action':jnp.tile(jnp.repeat(jnp.arange(3,6,dtype=jnp.int32),q),7*p)}
            _,(last_feat,_)=agent.model.dyn.imagine(branches,endings,1,False,single=True)
            last=agent.model.rew(agent.model.feat2tensor(last_feat),1).pred().reshape(7,p,3,q)
            # Only information-gathering actions can have a subsequent diagnosis.
            return combine_values(first,cont,last,jnp.arange(7)<3)
        self.predict=jax.jit(lambda params,dyn,key:lookahead(params,dyn,seed=key,create=False,modify=False)[1])

    def __call__(self, carry, obs, mode='eval'):
        import jax
        if mode!='eval':raise ValueError('Planner is frozen and evaluation-only')
        if len(obs['is_first'])!=1:raise ValueError('Planner supports one environment')
        if obs['is_first'][0]:self.waited=False
        carry,_,outs=self.agent.policy(carry,obs,mode='eval')
        with jax.transfer_guard('allow'):
            dyn=self.agent._stack(carry)[1]
            values=np.asarray(self.predict(self.agent.params,dyn,jax.random.PRNGKey(self.seed+self.calls))).copy()
            self.calls+=1
            action=choose_action(values,obs['vector'][0],self.waited)
            self.waited=self.waited or action==0
            previous={**carry[-1],'action':[jax.device_put(np.int32(action),carry[-1]['action'][0].sharding)]}
            carry=(*carry[:-1],previous)
        return carry,{'action':np.array([action],np.int32)},outs

    policy=__call__

    def init_policy(self,n):return self.agent.init_policy(n)
