"""Supervised temporal MLP: same telemetry and labels, no dynamics objective."""
import argparse,json,time
from pathlib import Path
import numpy as np
from benchmark.dreamer import sequence

def main():
    p=argparse.ArgumentParser()
    for key in ('root','data','train-answers','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    import jax,jax.numpy as jnp,optax
    start=time.monotonic();out=Path(a.out);out.mkdir(exist_ok=False)
    meta=json.loads((Path(a.root)/'metadata.json').read_text());cols=meta['columns'];classes=meta['methods']['dreamer_state']['classes']
    rows=json.loads((Path(a.data)/'manifest.json').read_text());assert all(r['split'] in ('train','test') for r in rows)
    truth={r['id']:r for r in json.loads(Path(a.train_answers).read_text())}
    assert set(truth)=={r['id'] for r in rows if r['split']=='train'}
    assert all(r['split']=='train' for r in truth.values())
    train=np.array([r['split']=='train' for r in rows])
    x=np.array([sequence(Path(a.data)/(r['id']+'.npz'),cols).reshape(-1) for r in rows]);assert np.isfinite(x).all()
    mean=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-6]=1
    x=(x-mean)/scale
    y=np.array([classes.index(truth[r['id']]['service']) for r in rows if r['split']=='train'])
    rng=np.random.default_rng(a.seed)
    params=[jnp.array(rng.normal(size=(x.shape[1],128))*np.sqrt(2/x.shape[1]),dtype=jnp.float32),jnp.zeros(128),jnp.array(rng.normal(size=(128,len(classes)))*np.sqrt(1/128),dtype=jnp.float32),jnp.zeros(len(classes))]
    def logits(p,z):return jax.nn.relu(z@p[0]+p[1])@p[2]+p[3]
    def loss(p,z,y):return optax.softmax_cross_entropy_with_integer_labels(logits(p,z),y).mean()
    opt=optax.adamw(1e-3,weight_decay=1e-4);state=opt.init(params)
    @jax.jit
    def step(p,state,z,y):
        value,grad=jax.value_and_grad(loss)(p,z,y);updates,state=opt.update(grad,state,p)
        return optax.apply_updates(p,updates),state,value
    xt=jnp.array(x[train]);yt=jnp.array(y)
    for i in range(32000):
        ix=rng.integers(0,len(y),16);params,state,value=step(params,state,xt[ix],yt[ix])
    scores=np.asarray(logits(params,jnp.array(x)));assert np.isfinite(scores).all()
    ranks=np.argsort(-scores,axis=1,kind='stable')
    (out/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],ranks=[classes[j] for j in ranks[i]]) for i,r in enumerate(rows)]))
    np.savez_compressed(out/'model.npz',mean=mean,scale=scale,**{'p'+str(i):np.asarray(v) for i,v in enumerate(params)})
    (out/'summary.json').write_text(json.dumps(dict(seed=a.seed,seconds=time.monotonic()-start,updates=32000,batch_size=16,hidden_units=128,parameters=sum(v.size for v in params),final_loss=float(value),test_labels_loaded=False,scope='Same incident labels and telemetry; capacity and compute not matched to Dreamer.')))
    print('SUPERVISED_BASELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
