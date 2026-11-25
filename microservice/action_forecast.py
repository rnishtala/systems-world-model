"""Prior-only repair outcome inference from the action-conditioned Dreamer checkpoint."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from microservice.action_model import VERSION,WIDTH,frame,make_agent
from microservice.action_outcomes import ACTIONS,TARGETS
from microservice.temporal_calibration import FEATURES

PARTICLES=16
RNG_SEED=777

def prefix_frames(observations):
    h=np.asarray(observations,dtype=np.float32)
    if h.shape!=(4,len(FEATURES)) or not np.isfinite(h).all() or (h<0).any():raise ValueError('Expected four finite nonnegative telemetry frames')
    # Deliberately has no outcome argument. Frame() cannot read outcomes at t<4.
    return np.stack([frame({'observations':h},t) for t in range(4)])

def physical_outcomes(values,symlog=True,xp=np):
    values=xp.asarray(values)
    logvalues=xp.sign(values)*xp.expm1(xp.abs(values)) if symlog else values
    return xp.expm1(logvalues)

def verify_source(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'summary.json').read_text());s=s.get('action_dreamer',s)
    if m['version']!=VERSION or s.get('status')!='training_complete' or s.get('version')!=VERSION:raise ValueError('Expected completed repair macro checkpoint')
    digest=hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest()
    if digest!=m['training_sha256'] or digest!=s['training_sha256']:raise ValueError('Training identity mismatch')
    if m['features']!=FEATURES or m['targets']!=list(TARGETS) or m['actions']!=list(ACTIONS):raise ValueError('Model schema mismatch')
    return m

def checkpoint_hash(root):
    root=Path(root)/'final';files=sorted(p for p in root.rglob('*') if p.is_file())
    if not files:raise ValueError('Missing final checkpoint')
    h=hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode()+b'\0');h.update(p.read_bytes())
    return h.hexdigest()

def prior_outcomes(inner,h,action,xp):
    enc,dyn,dec,prev=inner.init_policy(PARTICLES)
    noop=2*xp.eye(4,dtype=xp.float32)[0]-1
    for t in range(4):
        reset=xp.full((PARTICLES,),t==0)
        obs={'vector':xp.repeat(h[t][None],PARTICLES,axis=0)}
        enc,_,tokens=inner.enc(enc,obs,reset,training=False,single=True)
        dyn,_,feat=inner.dyn.observe(dyn,tokens,prev,reset,training=False,single=True)
        dec,_,_=inner.dec(dec,feat,reset,training=False,single=True)
        prev={'action':xp.repeat(noop[None],PARTICLES,axis=0)}
    # Encode the proposed intervention only after assimilating the prefix.
    proposed={'action':xp.repeat(action[None],PARTICLES,axis=0)}
    _,(prior,_)=inner.dyn.imagine(dyn,proposed,1,training=False,single=True)
    _,_,prediction=inner.dec(dec,prior,xp.zeros((PARTICLES,),bool),training=False,single=True)
    raw=prediction['vector'].pred()[:,len(FEATURES):len(FEATURES)+4]
    values=physical_outcomes(raw,inner.dec.symlog,xp)
    return values.mean(0),xp.isfinite(values).all()

class Forecaster:
    def __init__(self,root):
        root=Path(root);verify_source(root);self.checkpoint_sha256=checkpoint_hash(root)
        from microservice.configuration import elements
        import jax,jax.numpy as jnp,ninjax as nj
        config=elements.Config.load(elements.Path(str(root/'train/config.yaml'))).update(logdir='/tmp/action-forecast')
        self.agent=make_agent(config,root/'train.jsonl')
        checkpoint=elements.Checkpoint(elements.Path(str(root/'final')));checkpoint.agent=self.agent;checkpoint.load(keys=['agent'])
        inner=self.agent.model;self.params=self.agent.params;self.jax=jax
        @nj.pure
        def run(h,action):
            return prior_outcomes(inner,h,action,jnp)
        self.run=jax.jit(lambda params,h,action,key:run(params,h,action,seed=key,create=False,modify=False)[1])
    def __call__(self,observations,action):
        if action not in ACTIONS:raise ValueError('Unknown proposed action')
        h=prefix_frames(observations);a=2*np.eye(4,dtype=np.float32)[ACTIONS.index(action)]-1
        with self.jax.transfer_guard('allow'):
            values,finite=self.run(self.params,h,a,self.jax.random.PRNGKey(RNG_SEED))
            values=np.asarray(values);finite=bool(np.asarray(finite))
        if not finite or values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('Nonfinite outcome forecast')
        values=np.maximum(values,0);values[0]=min(values[0],1)
        return values.tolist()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    source=Path(a.source);meta=verify_source(source);before=checkpoint_hash(source)
    model=Forecaster(source);rows=[json.loads(l) for l in (source/'train.jsonl').read_text().splitlines()]
    # Fixed smoke selection from training only, no test scoring or sample selection by error.
    selected=[next(r for r in rows if r['action']==action) for action in ACTIONS]
    predictions=[]
    for row in selected:
        if row['split']!='train':raise ValueError('Training-only inference smoke')
        result={action:model(row['observations'],action) for action in ACTIONS}
        repeat=model(row['observations'],row['action'])
        if not np.allclose(repeat,result[row['action']],rtol=1e-6,atol=1e-6):raise ValueError('Forecast is not repeatable')
        predictions.append(result)
    after=checkpoint_hash(source)
    if before!=after:raise ValueError('Source checkpoint changed')
    if verify_source(source)['training_sha256']!=meta['training_sha256']:raise ValueError('Source training identity changed')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'smoke-predictions.json').write_text(json.dumps(predictions))
    summary=dict(status='inference_smoke_complete',source_run=source.parent.name,version=VERSION,training_sha256=meta['training_sha256'],checkpoint_sha256=before,checkpoint_unchanged=True,observed_prefixes=4,proposed_actions=4,forecasts=16,particles=PARTICLES,rng_seed=RNG_SEED,all_finite=True,repeatable=True,scope='Checkpoint reload and prior-only inference smoke on training prefixes; no accuracy, counterfactual validity, or repair-policy claim')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('ACTION_DREAMER_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
