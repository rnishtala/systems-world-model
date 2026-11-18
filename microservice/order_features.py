"""Offline prefix-tree replay for all check orders; no fault/control API calls."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from microservice.environment import Incident
from microservice.latent_probe import digest
ORDERS=list(itertools.permutations((1,2,3)))
PREFIXES=[()]+[p for n in range(1,4) for p in itertools.permutations((1,2,3),n)]

def observed(row,prefix):
    # Truth is immaterial to inspection observations and is deliberately omitted.
    sim=Incident(row['frames'],0,compositional=True);values=[sim.observe()]
    for action in prefix:values.append(sim.step(action)[0])
    return np.asarray(values)

def force_action(carry,action,put=None):
    if put is None:
        import jax
        put=jax.device_put
    return (*carry[:-1],{**carry[-1],'action':[put(np.int32(action),ref.sharding) for ref in carry[-1]['action']]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source)
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==80 and all(r['seed']==8201 for r in rows)
    sh=hashlib.sha256(source.read_bytes()).hexdigest();checkpoint=Path(a.root)/'final-51';ch=digest(checkpoint)
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    import jax
    model=Model(a.root,51,'inspect-mechanisms');audited=audit(model);result={};verified_actions=0
    for order in ORDERS:
        key=''.join(map(str,order))
        for name in ('raw','latent'):
            for t in range(4):result[f'{name}_{key}_{t}']=[]
    for i,row in enumerate(rows):
        cache={}
        for prefix in PREFIXES:
            sequence=observed(row,prefix);vector=sequence[-1];first=not prefix
            carry=model.policy.init_policy(1) if first else force_action(cache[prefix[:-1]][0],prefix[-1])
            if not first:
                with jax.transfer_guard('allow'):
                    assert all(np.all(np.asarray(ref)==prefix[-1]) for ref in carry[-1]['action'])
                verified_actions+=1
            obs=dict(vector=np.array([vector],np.float32),reward=np.array([0. if first else -.2],np.float32),is_first=np.array([first]),is_last=np.array([False]),is_terminal=np.array([False]))
            if model.supervised:obs['mechanism_target']=np.zeros((1,3),np.float32)
            carry,_,_=model.policy.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                dyn=model.agent._stack(carry)[1]
                state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            cache[prefix]=(carry,sequence.reshape(-1),state)
        for order in ORDERS:
            key=''.join(map(str,order))
            for t in range(4):
                _,raw,latent=cache[order[:t]]
                result[f'raw_{key}_{t}'].append(raw);result[f'latent_{key}_{t}'].append(latent)
        if i%10==0:print('ORDER_EXTRACT',i+1,flush=True)
    np.savez_compressed(out/'features.npz',**{k:np.array(v) for k,v in result.items()})
    assert digest(checkpoint)==ch and hashlib.sha256(source.read_bytes()).hexdigest()==sh
    assert verified_actions==80*15
    identity=dict(n=80,scenario_seed=8201,source_sha256=sh,checkpoint_sha256=ch,source_unchanged=True,checkpoint_unchanged=True,orders=ORDERS,prefix_nodes_per_incident=16,verified_previous_actions=verified_actions,shared_prefixes_reused=True,label_audit=audited,features_sha256=hashlib.sha256((out/'features.npz').read_bytes()).hexdigest())
    (out/'identity.json').write_text(json.dumps(identity,indent=2));print('ORDER_EXTRACTION_COMPLETE',json.dumps(identity),flush=True)
if __name__=='__main__':main()
