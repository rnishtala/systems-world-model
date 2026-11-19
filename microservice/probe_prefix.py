"""Offline posterior states after each measured prefix, no future telemetry."""
import numpy as np
from microservice.probe_target import frames,vector,next_action
def extract_prefixes(model,rows):
    import jax
    from scripts.evaluate_ledger_probe import features as raw_features
    result={f"{name}_{t}":[] for name in ("latent","history","summary_history") for t in range(8)}
    for i,row in enumerate(rows):
        carry=model.policy.init_policy(1);sequence=[];fs=frames(row)
        for t in range(8):
            v=vector(row,t);sequence.append(v)
            with jax.transfer_guard('allow'):
                if t:
                    expected=2*np.array([float(fs[t]['target']=='ledger'),fs[t]['requests']/12],np.float32)-1
                    for a in carry[-1]['action']:np.testing.assert_allclose(np.asarray(a),expected,atol=1e-6)
            obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),mechanism_target=np.zeros((1,3),np.float32))
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                action=2*next_action(row,t)-1
                carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(action,ref.sharding) for ref in carry[-1]['action']]})
                dyn=model.agent._stack(carry)[1]
                state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            history=np.array(sequence).reshape(-1)
            # Only observed prefixes: per-target/load means, counts, and availability.
            summary=[]
            for target in (0.,1.):
                for load in (.25,.75):
                    seen=[u for u in sequence if u[4]==target and np.isclose(u[5],load)]
                    summary.extend(np.mean(np.array(seen)[:,:4],axis=0) if seen else np.zeros(4))
                    summary.extend([len(seen),float(bool(seen))])
            result[f'latent_{t}'].append(state)
            result[f'history_{t}'].append(history)
            result[f'summary_history_{t}'].append(np.r_[summary,history])
        if i%8==0:print('PROBE_EXTRACT',i,flush=True)
    return {k:np.asarray(v) for k,v in result.items()}


def main():
    import hashlib,json
    from pathlib import Path
    from microservice.agent import Model
    from microservice.latent_probe import digest
    from microservice.auxiliary_audit import audit
    root=Path('/results/probe-target-model-v1/models')
    before=digest(root/'final-91');model=Model(str(root),91,'native')
    rows=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()]
    assert len(rows)==64 and {r['seed'] for r in rows}=={9101,9103}
    out=Path('evaluation');out.mkdir(exist_ok=False)
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    xs=extract_prefixes(model,rows)
    assert all(np.isfinite(x).all() for x in xs.values())
    np.savez_compressed(out/'prefix-features.npz',**xs)
    assert before==digest(root/'final-91')
    (out/'extraction.json').write_text(json.dumps(dict(checkpoint_sha256=before,checkpoint_unchanged=True,n=64,
        shapes={k:list(v.shape) for k,v in xs.items()},train_sha256=hashlib.sha256(Path('collection/train.jsonl').read_bytes()).hexdigest()),indent=2))
    print('PROBE_PREFIX_COMPLETE',flush=True)
if __name__=='__main__':main()
