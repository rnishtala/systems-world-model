"""Read-only replay of frozen checkpoint: decoder units and transition audit."""
import json,hashlib
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,ninjax as nj
from microservice.agent import Model
from microservice.workload import vector,next_load
from microservice.latent_probe import digest
from microservice.decoder_units import observation_prediction
ROOT=Path('/results/lab-workload-v2');OUT=Path('/tmp/forecast-audit');OUT.mkdir(exist_ok=False)
CHANNELS=[0,3,4,5]

def main():
    before=digest(ROOT/'models/final-81');model=Model(str(ROOT/'models'),81,'native');inner=model.agent.model
    assert inner.dec.symlog is True
    @nj.pure
    def decode(dyn,action):
        _,_,post=inner.dec(inner.dec.initial(16),dyn,jnp.zeros(16,bool),training=False,single=True)
        _,(feat,_)=inner.dyn.imagine(dyn,{'action':jnp.full((16,1),action,jnp.float32)},1,False,single=True)
        _,_,prior=inner.dec(inner.dec.initial(16),feat,jnp.zeros(16,bool),training=False,single=True)
        p=prior['vector'].pred();q=post['vector'].pred()
        return p.mean(0),observation_prediction(p,True,jnp).mean(0),q.mean(0),observation_prediction(q,True,jnp).mean(0)
    run=jax.jit(lambda params,dyn,action,key:decode(params,dyn,action,seed=key,create=False,modify=False)[1])
    summary={'checkpoint_before':before,'decoder_symlog':True,'channels':['checkout','payment_self','ledger','payment_downstream'],'splits':{}}
    for split in ('train','validation'):
        path=ROOT/('collection/train.jsonl' if split=='train' else 'evaluation/validation.jsonl')
        rows=[json.loads(l) for l in path.read_text().splitlines()];raw=[];fixed=[];posterior=[];rawpost=[];current=[];actual=[];steps=[]
        for i,row in enumerate(rows):
            carry=model.policy.init_policy(1)
            for t in range(4):
                v=vector(row,t)
                with jax.transfer_guard('allow'):
                    if t:assert all(np.isclose(np.asarray(a).item(),2*row['frames'][t]['requests']/12-1) for a in carry[-1]['action'])
                obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]),mechanism_target=np.zeros((1,3),np.float32))
                carry,_,_=model.policy.policy(carry,obs,mode='eval')
                action=np.float32(2*next_load(row,t)-1)
                with jax.transfer_guard('allow'):
                    carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.array([action],np.float32),ref.sharding) for ref in carry[-1]['action']]})
                    if t<3:
                        dyn=model.agent._stack(carry)[1];key=jax.random.PRNGKey(8804+i*4+t)
                        r,f,q,p=[np.asarray(a)[CHANNELS] for a in run(model.agent.params,dyn,action,key)]
                        # Standalone decoding and production adapter must agree.
                        adapter=np.asarray(model.forecast(model.agent.params,dyn,action,key))[CHANNELS]
                        np.testing.assert_allclose(adapter,f,rtol=1e-5,atol=1e-6)
                        raw.append(r);fixed.append(f);rawpost.append(q);posterior.append(p);current.append(v[CHANNELS]);actual.append(vector(row,t+1)[CHANNELS]);steps.append(t)
            if i%32==0:print('AUDIT_PROGRESS',split,i,len(rows),flush=True)
        arrays={k:np.asarray(v) for k,v in dict(raw=raw,fixed=fixed,raw_posterior=rawpost,posterior=posterior,current=current,actual=actual,steps=steps).items()}
        saved=np.load(ROOT/f'evaluation/{split}-forecast.npz')
        np.testing.assert_allclose(arrays['raw'],saved['predicted'],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(arrays['actual'],saved['actual'])
        np.testing.assert_array_equal(arrays['current'],saved['persistence'])
        np.savez_compressed(OUT/f'{split}.npz',**arrays)
        a=arrays;metrics={}
        for name,pred,target in [('raw_forecast',a['raw'],a['actual']),('corrected_forecast',a['fixed'],a['actual']),('persistence',a['current'],a['actual']),('raw_posterior',a['raw_posterior'],a['current']),('corrected_posterior',a['posterior'],a['current'])]:
            err=(pred-target)**2;metrics[name]={'mse':float(err.mean()),'by_channel':err.mean(0).tolist(),'by_step':[float(err[a['steps']==t].mean()) for t in range(3)]}
        summary['splits'][split]=dict(incidents=len(rows),transitions=len(actual),raw_forecasts_reproduced=True,production_adapter_matches=True,metrics=metrics)
    summary['checkpoint_after']=digest(ROOT/'models/final-81');assert summary['checkpoint_after']==before
    summary['checkpoint_unchanged']=True
    summary['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('AUDIT_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
