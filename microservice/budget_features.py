"""Prefix inference using only actual measured frames and declared schedule."""
import numpy as np

def planned_action(row,t):
    if t>=7:return np.zeros(2,np.float32)
    target,n=row['schedule'][t+1]
    return np.array([float(target=='ledger'),n/12],np.float32)

def budget_vector(row,t):
    f=row['frames'][t];m=f['metrics'];upstream=f['target']=='checkout'
    assert [f['target'],f['requests']]==row['schedule'][t]
    assert f['successes']==f['requests']
    latency=[m['ledger']['latency_p95'],m['payment']['self_p95'] if upstream else 0.,m['checkout']['latency_p95'] if upstream else 0.,m['payment']['downstream_p95'] if upstream else 0.]
    if not np.isfinite(latency).all() or min(latency)<0:raise ValueError('Invalid latency')
    return np.array([*np.log1p(np.array(latency)/.1),float(not upstream),f['requests']/12,*planned_action(row,t),float(upstream),t/7,*([float(t>=7)]*3),m['ledger']['error_rate']],np.float32)

def extract_budget(model,rows):
    import jax
    result={f"{name}_{t}":[] for name in ("latent","history","summary_history") for t in range(len(rows[0]["frames"]))}
    for i,row in enumerate(rows):
        carry=model.policy.init_policy(1);sequence=[];fs=row["frames"]
        for t in range(len(fs)):
            v=budget_vector(row,t);sequence.append(v)
            with jax.transfer_guard('allow'):
                if t:
                    expected=2*np.array([float(fs[t]['target']=='ledger'),fs[t]['requests']/12],np.float32)-1
                    for a in carry[-1]['action']:np.testing.assert_allclose(np.asarray(a),expected,atol=1e-6)
            obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),mechanism_target=np.zeros((1,3),np.float32))
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                action=2*planned_action(row,t)-1
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
