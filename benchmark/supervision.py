"""Training-only service targets; excluded from policy, encoder and decoder."""
TARGET='root_cause_target'
def telemetry(obs):return {k:v for k,v in obs.items() if k!=TARGET}

def objective(base,diagnosis,auxiliary_only):
    return diagnosis if auxiliary_only else base+diagnosis

def make_agent(config,classes,auxiliary_only=False):
    from dreamerv3 import agent as module,main
    import embodied.jax.nets as nn
    import jax,jax.numpy as jnp
    original=module.Agent
    class DiagnosisAgent(original):
        def __init__(self,obs_space,act_space,config):
            super().__init__(telemetry(obs_space),act_space,config)
            self.obs_space=obs_space
            self.diagnosis=nn.Linear(classes,name='diagnosis')
            self.modules.append(self.diagnosis)
        def policy(self,carry,obs,mode='train'):
            return super().policy(carry,telemetry(obs),mode)
        def loss(self,carry,obs,prevact,training):
            target=obs[TARGET];clean=telemetry(obs)
            loss,(carry,entries,outs,metrics)=super().loss(carry,clean,prevact,training)
            logits=self.diagnosis(self.feat2tensor(outs['repfeat'])).astype(jnp.float32)
            valid=target.sum(-1)>0;count=jnp.maximum(valid.sum(),1)
            ce=-(target*jax.nn.log_softmax(logits,-1)).sum(-1)
            aux=(ce*valid).sum()/count
            metrics=dict(metrics,diagnosis_loss=aux,diagnosis_accuracy=((logits.argmax(-1)==target.argmax(-1))*valid).sum()/count)
            total=objective(loss,aux,auxiliary_only)
            metrics["optimized_loss"]=total
            metrics["world_model_objective_enabled"]=jnp.asarray(not auxiliary_only,jnp.float32)
            return total,(carry,entries,outs,metrics)
    module.Agent=DiagnosisAgent
    try:return main.make_agent(config)
    finally:module.Agent=original

def audit(agent,dimensions,classes):
    import numpy as np,jax,ninjax as nj
    inner=agent.model
    assert TARGET not in inner.enc.obs_space and TARGET not in inner.dec.obs_space
    assert inner.opt.modules is inner.modules and inner.diagnosis in inner.modules
    @nj.pure
    def run(obs):return inner.policy(inner.init_policy(1),obs,mode='eval')
    obs=dict(vector=np.zeros((1,dimensions),np.float32),reward=np.zeros(1,np.float32),is_first=np.ones(1,bool),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool))
    with jax.transfer_guard('allow'):
        result=[run(agent.params,{**obs,TARGET:np.full((1,classes),v,np.float32)},seed=jax.random.PRNGKey(777),create=False,modify=False)[1] for v in (0,1)]
        for a,b in zip(jax.tree.leaves(result[0]),jax.tree.leaves(result[1])):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    return dict(encoder_excludes_labels=True,decoder_excludes_labels=True,policy_identical_for_opposite_labels=True,auxiliary_head_optimized=True)
