"""Frozen Dreamer one-step prior forecasts; future observations never enter inputs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.probe_target import vector
from microservice.validate import read_rows,sha

CUTS=(4,5,6,7)
PENALTIES=(.01,.1,1.,10.,100.)

def history(row,cut):
    if cut not in CUTS:raise ValueError('Unsupported observation prefix')
    return np.stack([vector(row,t) for t in range(cut)])

def design(h):
    padded=np.zeros((7,14));padded[:len(h)]=h
    return np.r_[padded.ravel(),np.arange(7)<len(h)]

def controls(h):
    action=h[-1,6:8]
    matching=[v[0] for v in h if np.allclose(v[4:6],action,atol=1e-6)]
    return dict(persistence=float(h[-1,0]),matching_probe=float(matching[-1] if matching else h[-1,0]))

def examples(rows):
    hs=[history(row,cut) for row in rows for cut in CUTS]
    x=np.stack([design(h) for h in hs])
    y=np.array([vector(row,cut)[0] for row in rows for cut in CUTS])
    groups=np.repeat([row['repeat']%4 for row in rows],len(CUTS))
    return hs,x,y,groups

def fit(x,y,penalty):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean()
    weight=np.linalg.solve(z.T@z+penalty*np.eye(z.shape[1]),z.T@(y-bias))
    return dict(mean=mean,scale=scale,weight=weight,bias=bias)

def predict(model,x):return ((x-model['mean'])/model['scale'])@model['weight']+model['bias']

def prepare(data,out):
    rows=read_rows(data);hs,x,y,groups=examples(rows)
    if set(groups)!={0,1,2,3}:raise ValueError('Need all four training groups')
    cv=[]
    for penalty in PENALTIES:
        pred=np.empty(len(y))
        for g in range(4):
            held=groups==g;pred[held]=predict(fit(x[~held],y[~held],penalty),x[held])
        cv.append(float(abs(pred-y).mean()))
    i=int(np.argmin(cv));model=fit(x,y,PENALTIES[i])
    errors={k:float(np.mean([abs(controls(h)[k]-v) for h,v in zip(hs,y)])) for k in controls(hs[0])}
    errors['regression']=cv[i]
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'regression.npz',**model)
    frozen=dict(utc=time.time(),train_sha256=sha(data),training_seeds=sorted({v['seed'] for v in rows}),cuts=list(CUTS),penalty=PENALTIES[i],cv_mae=errors,selected_control=min(errors,key=errors.get),regression_sha256=sha(out/'regression.npz'),primary_metric='mean absolute error in log1p(ledger latency / 0.1 seconds)',particles=16)
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    return frozen

def observation_prediction(values,symlog=True,xp=np):
    """Pinned symlog_mse.pred() returns transformed values; invert per particle."""
    values=xp.asarray(values)
    return xp.sign(values)*xp.expm1(xp.abs(values)) if symlog else values

class Forecaster:
    def __init__(self,model,diagnostic=False):
        import jax,jax.numpy as jnp,ninjax as nj
        inner=model.agent.model;self.params=model.agent.params;self.jax=jax
        @nj.pure
        def run(h):
            enc,dyn,dec,prev=inner.init_policy(16)
            for t in range(h.shape[0]):
                reset=jnp.full((16,),t==0)
                obs={'vector':jnp.repeat(h[t][None],16,axis=0)}
                enc,_,tokens=inner.enc(enc,obs,reset,training=False,single=True)
                dyn,_,feat=inner.dyn.observe(dyn,tokens,prev,reset,training=False,single=True)
                dec,_,reconstruction=inner.dec(dec,feat,reset,training=False,single=True)
                prev={'action':jnp.repeat((2*h[t,6:8]-1)[None],16,axis=0)}
            # Advance only with the known next action; no next-observation tokens.
            _,(prior,_)=inner.dyn.imagine(dyn,prev,1,training=False,single=True)
            _,_,pred=inner.dec(dec,prior,jnp.zeros((16,),bool),training=False,single=True)
            raw=pred['vector'].pred()[:,0]
            forecast=observation_prediction(raw,inner.dec.symlog,jnp).mean()
            if diagnostic:
                reconstructed=observation_prediction(reconstruction['vector'].pred()[:,0],inner.dec.symlog,jnp).mean()
                return jnp.stack([reconstructed,forecast,raw.mean()])
            return forecast
        self.diagnostic=diagnostic
        self.run=jax.jit(lambda params,h,key:run(params,h,seed=key,create=False,modify=False)[1])
    def __call__(self,h):
        # Fixed RNG and particle budget; no tuning from test outcomes.
        with self.jax.transfer_guard('allow'):
            value=self.run(self.params,np.asarray(h,np.float32),self.jax.random.PRNGKey(777))
            return np.asarray(value) if self.diagnostic else float(value)

def summarize(y,pred,selected,n):
    y=np.asarray(y);errors={k:abs(np.asarray(v)-y).reshape(n,len(CUTS)).mean(1) for k,v in pred.items()}
    if any(not np.isfinite(v).all() for v in pred.values()):raise ValueError('Nonfinite forecast')
    metrics={}
    for k,v in pred.items():
        with np.errstate(over='raise',invalid='raise'):
            ms=100*np.expm1(np.maximum(0,np.asarray(v)));actual=100*np.expm1(y)
        metrics[k]=dict(log_mae=float(errors[k].mean()),milliseconds_mae=float(abs(ms-actual).mean()),by_prefix={str(c):float(abs(np.asarray(v)-y).reshape(n,len(CUTS))[:,i].mean()) for i,c in enumerate(CUTS)})
    delta=errors['dreamer']-errors[selected]
    rng=np.random.default_rng(20261121);means=delta[rng.integers(0,n,(2000,n))].mean(1)
    ci=np.quantile(means,[.025,.975]).tolist();base=metrics[selected]['log_mae'];new=metrics['dreamer']['log_mae']
    return dict(incidents=n,predictions=len(y),methods=metrics,selected_control=selected,relative_reduction=1-new/base if base>0 else None,paired_incident_bootstrap_95=ci,improvement_gate=bool(base>0 and new<=.9*base and ci[1]<0),scope='One-step prediction under measured schedules; not unseen-action planning or remediation',primary_metric='log_mae')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','evaluate']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--models',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--readouts');a=p.parse_args()
    from microservice.readout import digest
    from microservice.model import Model
    checkpoint=Path(a.models)/f'final-{a.seed}';before=digest(checkpoint)
    rows=read_rows(a.data)
    if a.phase=='prepare':
        meta=json.loads((Path(a.models)/f'training-{a.seed}.json').read_text())
        if meta['calibration_sha256']!=sha(a.data):raise ValueError('Use exact source training data')
        frozen=prepare(a.data,a.out)
        model=Model(a.models,a.seed);forecaster=Forecaster(model)
        smoke=[forecaster(history(rows[0],cut)) for cut in CUTS]
        if not np.isfinite(smoke).all():raise ValueError('Invalid prior forecast')
        frozen.update(checkpoint_sha256=before,smoke_predictions=smoke,utc=time.time())
        (Path(a.out)/'frozen.json').write_text(json.dumps(frozen,indent=2))
    else:
        from microservice.validate import check_test
        if not a.readouts:raise ValueError('Frozen controls required')
        frozen=json.loads((Path(a.readouts)/'frozen.json').read_text());check_test(rows,frozen)
        if before!=frozen['checkpoint_sha256'] or sha(Path(a.readouts)/'regression.npz')!=frozen['regression_sha256']:raise ValueError('Frozen artifacts changed')
        with np.load(Path(a.readouts)/'regression.npz',allow_pickle=False) as f:regression=dict(f)
        hs,x,y,_=examples(rows);pred={k:[controls(h)[k] for h in hs] for k in controls(hs[0])};pred['regression']=predict(regression,x).tolist()
        model=Model(a.models,a.seed);forecaster=Forecaster(model);pred['dreamer']=[]
        for i,h in enumerate(hs):
            pred['dreamer'].append(forecaster(h))
            if i%32==0:print('FORECAST_PROGRESS',i,'/',len(hs),flush=True)
        out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
        if sha(Path(a.readouts)/'regression.npz')!=frozen['regression_sha256']:raise ValueError('Controls changed during evaluation')
        summary=summarize(y,pred,frozen['selected_control'],len(rows))
        (out/'predictions.json').write_text(json.dumps(dict(test_sha256=sha(a.data),target=y.tolist(),methods=pred,cuts=list(CUTS))))
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if digest(checkpoint)!=before:raise ValueError('Checkpoint changed')
    print('FORECAST_COMPLETE',a.phase,flush=True)

if __name__=='__main__':main()
