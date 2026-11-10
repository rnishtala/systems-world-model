"""Actual upstream checkpoint and label-invariance audit."""
import argparse,json
from pathlib import Path
import numpy as np

def audit(model):
    import jax,ninjax as nj
    from microservice.supervision import TARGET
    inner=model.agent.model
    assert TARGET not in inner.enc.obs_space and TARGET not in inner.dec.obs_space
    assert inner.opt.modules is inner.modules and inner.mechanisms in inner.modules
    @nj.pure
    def run(obs):
        carry=inner.init_policy(1)
        return inner.policy(carry,obs,mode='eval')
    base=dict(vector=np.zeros((1,14),np.float32),reward=np.zeros(1,np.float32),is_first=np.ones(1,bool),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool))
    with jax.transfer_guard('allow'):
        results=[run(model.agent.params,{**base,TARGET:np.full((1,3),v,np.float32)},seed=jax.random.PRNGKey(777),create=False,modify=False)[1] for v in (0,1)]
        for a,b in zip(jax.tree.leaves(results[0]),jax.tree.leaves(results[1])):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    return dict(checkpoint_loaded=True,encoder_excludes_labels=True,decoder_excludes_labels=True,policy_identical_for_opposite_labels=True,auxiliary_head_optimized=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from microservice.agent import Model
    result=audit(Model(a.root,21,'inspect-mechanisms'))
    Path(a.out).write_text(json.dumps(result,indent=2));print('AUDIT_PASSED',result,flush=True)
