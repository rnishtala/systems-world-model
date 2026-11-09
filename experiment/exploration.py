"""Training-only coverage: inspect both sensors, then choose a random diagnosis.

This uses no hidden fault labels and supplies no expert diagnosis. Evaluation
always uses the unmodified learned policy. The recurrent previous-action state
must match the action that was actually executed, including exploration.
"""
import numpy as np


class InspectionCoverage:
    def __init__(self, policy, seed, fraction=.5):
        if not 0 <= fraction <= 1:
            raise ValueError('fraction must be between zero and one')
        self.policy=policy
        self.rng=np.random.default_rng(seed)
        self.fraction=fraction
        self.guided=None
        self.reverse=None
        self.phase=None
        self.stats={'guided_episodes':0,'policy_episodes':0,'overridden_actions':0}

    def __call__(self, carry, obs, mode='train'):
        carry,acts,outs=self.policy(carry,obs,mode=mode)
        if mode!='train':
            return carry,acts,outs
        n=len(obs['is_first'])
        if self.guided is None:
            self.guided=np.zeros(n,bool);self.reverse=np.zeros(n,bool)
            self.phase=np.zeros(n,np.int32)
        actions=acts['action'].copy()
        for i,first in enumerate(obs['is_first']):
            if first:
                self.guided[i]=self.rng.random()<self.fraction
                self.reverse[i]=self.rng.random()<.5
                self.phase[i]=0
                key='guided_episodes' if self.guided[i] else 'policy_episodes'
                self.stats[key]+=1
            if self.guided[i] and not obs['is_last'][i]:
                phase=self.phase[i]
                actions[i]=((2 if self.reverse[i] else 1) if phase==0 else
                            (1 if self.reverse[i] else 2) if phase==1 else
                            int(self.rng.integers(3,6)))
                self.phase[i]+=1
                self.stats['overridden_actions']+=1
        # Pinned upstream uses a tuple of state trees, with lists for batch items.
        previous_actions=[]
        for i,a in enumerate(actions):
            value=np.asarray(a,np.int32)
            if hasattr(carry[-1]['action'][i], 'sharding'):
                import jax
                value=jax.device_put(value,carry[-1]['action'][i].sharding)
            previous_actions.append(value)
        previous={**carry[-1],'action':previous_actions}
        carry=(*carry[:-1],previous)
        return carry,{**acts,'action':actions},outs
