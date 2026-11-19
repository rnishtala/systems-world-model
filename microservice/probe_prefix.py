"""Offline posterior states after each measured prefix, no future telemetry."""
import numpy as np
from microservice.probe_target import frames,vector,next_action
def extract_prefixes(model,rows):
    import jax
    result={f"{name}_{t}":[] for name in ("latent","history") for t in range(8)}
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
            result[f'latent_{t}'].append(state)
            result[f'history_{t}'].append(history)
        if i%8==0:print('PROBE_EXTRACT',i,flush=True)
    return {k:np.asarray(v) for k,v in result.items()}
