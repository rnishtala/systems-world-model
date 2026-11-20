"""Training-only mechanism targets; never encoder inputs or reconstruction targets."""
import numpy as np
TARGET = 'mechanism_target'

def telemetry(obs):
    return {k: v for k, v in obs.items() if k != TARGET}

def diagnosis_mask(vector, is_last):
    """Supervise six and eight observed windows; never the terminal duplicate."""
    progress = vector[..., 9]
    return (((abs(progress - 5 / 7) < 1e-5) | (abs(progress - 1) < 1e-5))
            & ~is_last)

def make_agent(config):
    # Import only after microservice.configuration sets the pinned upstream import path.
    from dreamerv3 import agent as module, main
    import embodied.jax.nets as nn
    import jax.numpy as jnp
    import jax
    original = module.Agent
    class SupervisedAgent(original):
        def __init__(self, obs_space, act_space, config):
            super().__init__(telemetry(obs_space), act_space, config)
            self.obs_space = obs_space  # Replay retains targets; encoder/decoder do not.
            self.mechanisms = nn.Linear(3, name='mechanisms')
            self.modules.append(self.mechanisms)  # Optimizer keeps this same list.
        def policy(self, carry, obs, mode='train'):
            return super().policy(carry, telemetry(obs), mode)
        def loss(self, carry, obs, prevact, training):
            target = obs[TARGET]
            clean = telemetry(obs)
            loss, (carry, entries, outs, metrics) = super().loss(carry, clean, prevact, training)
            logits = self.mechanisms(self.feat2tensor(outs['repfeat'])).astype(jnp.float32)
            # Six-window posterior gets a direct training signal, alongside
            # the complete eight-window posterior. Labels never enter inputs.
            valid = diagnosis_mask(clean['vector'], clean['is_last'])
            count = jnp.maximum(valid.sum(), 1)
            bce = jax.nn.softplus(logits) - target * logits
            auxiliary = (bce.mean(-1) * valid).sum() / count
            metrics = dict(metrics, mechanism_loss=auxiliary,
                mechanism_exact=((jnp.all((logits > 0) == (target > .5), -1) * valid).sum() / count),
                mechanism_count=valid.sum())
            return loss + auxiliary, (carry, entries, outs, metrics)
    module.Agent = SupervisedAgent
    try: return main.make_agent(config)
    finally: module.Agent = original
