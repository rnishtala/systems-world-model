"""Training-only posterior reconstruction versus prior prediction diagnostic."""
import argparse,json,os
from pathlib import Path
import numpy as np
from microservice.action_forecast import Forecaster,verify_source,checkpoint_hash,physical_outcomes,PARTICLES,RNG_SEED
from microservice.action_model import frame,validate
from microservice.action_outcomes import ACTIONS,TARGETS,clip
from microservice.temporal_calibration import FEATURES

def diagnostic(inner,frames,action,xp):
    enc,dyn,dec,prev=inner.init_policy(PARTICLES)
    repeat=lambda v:xp.repeat(v[None],PARTICLES,axis=0)
    noop=2*xp.eye(4,dtype=xp.float32)[0]-1
    for t in range(4):
        reset=xp.full((PARTICLES,),t==0)
        enc,_,tokens=inner.enc(enc,{'vector':repeat(frames[t])},reset,training=False,single=True)
        dyn,_,feat=inner.dyn.observe(dyn,tokens,prev,reset,training=False,single=True)
        dec,_,_=inner.dec(dec,feat,reset,training=False,single=True)
        prev={'action':repeat(noop)}
    proposed={'action':repeat(action)};reset=xp.zeros((PARTICLES,),bool)
    _,(prior,_)=inner.dyn.imagine(dyn,proposed,1,training=False,single=True)
    _,_,prior_pred=inner.dec(dec,prior,reset,training=False,single=True)
    # Outcome is supplied ONLY to the reconstruction branch, never the prior.
    _,_,tokens=inner.enc(enc,{'vector':repeat(frames[4])},reset,training=False,single=True)
    _,_,posterior=inner.dyn.observe(dyn,tokens,proposed,reset,training=False,single=True)
    _,_,post_pred=inner.dec(dec,posterior,reset,training=False,single=True)
    start=len(FEATURES);target=frames[4,start:start+4]
    transformed=xp.sign(target)*xp.log1p(xp.abs(target)) if inner.dec.symlog else target
    result={}
    for name,pred in [('prior',prior_pred),('reconstruction',post_pred)]:
        raw=pred['vector'].pred()[:,start:start+4];physical=physical_outcomes(raw,inner.dec.symlog,xp)
        result[name]=dict(raw_mean=raw.mean(0),physical_mean=physical.mean(0),transformed_mse=((raw-transformed)**2).mean(0),finite=xp.isfinite(physical).all())
    kl=inner.dyn._dist(posterior['logit']).kl(inner.dyn._dist(prior['logit']))
    result['kl']=kl.mean();result['kl_with_free_nats']=xp.maximum(kl,inner.dyn.free_nats).mean()
    return result

def summarize(results,outcomes):
    target=np.asarray(outcomes);summary={}
    for name in ('prior','reconstruction'):
        pred=np.asarray([r[name]['physical_mean'] for r in results]);raw=np.asarray([r[name]['raw_mean'] for r in results])
        mse=np.asarray([r[name]['transformed_mse'] for r in results])
        if not np.isfinite(pred).all() or not np.isfinite(raw).all() or not np.isfinite(mse).all() or not all(r[name]['finite'] for r in results):raise ValueError('Nonfinite diagnostic')
        bounded=clip(pred)
        summary[name]={k:dict(raw_mean=float(raw[:,j].mean()),physical_mean=float(pred[:,j].mean()),observed_mean=float(target[:,j].mean()),mae=float(abs(bounded[:,j]-target[:,j]).mean()),unclipped_mae=float(abs(pred[:,j]-target[:,j]).mean()),clipped_fraction=float((bounded[:,j]!=pred[:,j]).mean()),transformed_mse=float(mse[:,j].mean())) for j,k in enumerate(TARGETS)}
    for k in ('kl','kl_with_free_nats'):
        values=np.asarray([r[k] for r in results])
        if not np.isfinite(values).all():raise ValueError('Nonfinite KL')
        summary[k]=float(values.mean())
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    source=Path(args.source);meta=verify_source(source);before=checkpoint_hash(source)
    rows=validate([json.loads(l) for l in (source/'train.jsonl').read_text().splitlines()])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(source_run=source.parent.name,training_sha256=meta['training_sha256'],checkpoint_sha256=before,rows=len(rows),particles=PARTICLES,rng_seed=RNG_SEED,scope='Training-only diagnostic; posterior sees outcome and is NOT a forecast; no fitting or fresh faults'),indent=2))
    model=Forecaster(source)
    import jax,jax.numpy as jnp,ninjax as nj
    @nj.pure
    def run(frames,action):return diagnostic(model.agent.model,frames,action,jnp)
    compiled=jax.jit(lambda params,frames,action,key:run(params,frames,action,seed=key,create=False,modify=False)[1])
    results=[]
    for row in rows:
        frames=np.stack([frame(row,t) for t in range(5)]);action=2*np.eye(4,dtype=np.float32)[ACTIONS.index(row['action'])]-1
        with jax.transfer_guard('allow'):
            result=compiled(model.params,frames,action,jax.random.PRNGKey(RNG_SEED))
            result=jax.tree.map(lambda x:np.asarray(x).tolist(),result)
        # Separately verify diagnostic prior agrees with the established forecaster.
        expected=model(row['observations'],row['action'])
        if not np.allclose(clip(np.array(result['prior']['physical_mean'])),expected,rtol=1e-5,atol=1e-6):raise ValueError('Prior path mismatch')
        results.append(result)
    scores=summarize(results,[r['outcome'] for r in rows])
    if checkpoint_hash(source)!=before or verify_source(source)['training_sha256']!=meta['training_sha256']:raise ValueError('Source changed')
    (out/'diagnostics.json').write_text(json.dumps(results))
    summary=dict(**scores,training_trials=len(rows),training_profiles=len({r['group'] for r in rows}),checkpoint_unchanged=True,prior_matches_forecaster=True,scope='In-sample diagnostic; reconstruction sees the outcome; no held-out accuracy claim',kl_note='Final-transition posterior||prior KL; dyn/rep have equal forward values, different gradients; not a recorded training loss')
    (out/'summary.json').write_text(json.dumps(dict(action_diagnostic=summary,incidents=[]),indent=2));print('ACTION_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
