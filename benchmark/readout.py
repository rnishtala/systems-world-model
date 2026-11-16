"""Validation-only readout study; test incidents are rejected, not loaded."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from benchmark.dreamer import sequence

def ridge(x,y,penalty):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    return mean,scale,z.T@np.linalg.solve(z@z.T+np.eye(len(z))*penalty,y-bias),bias

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--answers',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    start=time.monotonic();root=Path(a.root);data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'manifest.json').read_text());assert all(r['split'] in ('train','validation') for r in manifest)
    truth={r['id']:r for r in json.loads(Path(a.answers).read_text())};assert set(truth)=={r['id'] for r in manifest};assert all(r['split']!='test' for r in truth.values())
    meta=json.loads((root/'metadata.json').read_text());cols=meta['columns'];services=meta['methods']['dreamer_state']['classes']
    seq={r['id']:sequence(data/(r['id']+'.npz'),cols) for r in manifest}
    sys.path.insert(0,os.environ['DREAMER_SOURCE'])
    import elements,jax
    from dreamerv3 import main as up
    class Shape:
        obs_space=dict(vector=elements.Space(np.float32,(len(cols)*2,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        act_space=dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,1))
        def close(self):pass
    c=elements.Config.load(elements.Path(root/'train/config.yaml')).update(logdir=str(out/'inference'))
    up.make_env=lambda config,index,**kw:up.wrap_env(Shape(),config)
    agent=up.make_agent(c);cp=elements.Checkpoint(elements.Path(root/'checkpoint'));cp.agent=agent;cp.load(keys=['agent'])
    features={name:[] for name in ('last_deter','post_mean_deter','post_mean_full','raw','mask_only')}
    # Average 16 posterior particles to reduce stochastic-state noise.
    for i,r in enumerate(manifest):
        carry=agent.init_policy(16);ds=[];fs=[]
        for t,v in enumerate(seq[r['id']]):
            obs=dict(vector=np.repeat(v[None],16,axis=0),reward=np.zeros(16,np.float32),is_first=np.full(16,t==0),is_last=np.full(16,t==19),is_terminal=np.full(16,t==19))
            carry,_,_=agent.policy(carry,obs,mode='eval')
            with jax.transfer_guard('allow'):
                dyn=agent._stack(carry)[1];d=np.asarray(dyn['deter']).reshape(16,-1).mean(0);s=np.asarray(dyn['stoch']).reshape(16,-1).mean(0)
            ds.append(d);fs.append(np.concatenate([d,s]))
        features['last_deter'].append(ds[-1]);features['post_mean_deter'].append(np.mean(ds[10:],axis=0));features['post_mean_full'].append(np.mean(fs[10:],axis=0));features['raw'].append(seq[r['id']].reshape(-1));features['mask_only'].append(seq[r['id']][:,len(cols):].reshape(-1))
        if i%25==0:print('READOUT',i,flush=True)
    train=np.array([r['split']=='train' for r in manifest]);val=~train
    y=np.array([[float(truth[r['id']]['service']==s) for s in services] for r in manifest]);targets=y.argmax(1)
    result={};rng=np.random.default_rng(1901)
    for name,rows in features.items():
        x=np.asarray(rows,dtype=np.float64);assert np.isfinite(x).all()
        for penalty in (1.,100.):
            for shuffled in (False,True):
                yy=y[train].copy()
                if shuffled:yy=yy[rng.permutation(len(yy))]
                mean,scale,w,bias=ridge(x[train],yy,penalty);scores=(x-mean)/scale@w+bias;assert np.isfinite(scores).all()
                ranks=np.argsort(-scores,axis=1,kind='stable');key=f'{name}_ridge{int(penalty)}'+('_shuffled' if shuffled else '')
                result[key]=dict(train_correct=int(sum(ranks[train,0]==targets[train])),validation_correct=int(sum(ranks[val,0]==targets[val])),validation_top3=int(sum(targets[i] in ranks[i,:3] for i in np.flatnonzero(val))),validation_n=int(val.sum()),dimensions=x.shape[1])
                (out/(key+'-predictions.json')).write_text(json.dumps([dict(id=r['id'],split=r['split'],ranks=[services[j] for j in ranks[i]]) for i,r in enumerate(manifest)]))
    np.savez_compressed(out/'features.npz',**{k:np.asarray(v) for k,v in features.items()})
    report=dict(results=result,seconds=time.monotonic()-start,particles=16,scope='Development validation only; no test incidents loaded; weights frozen; predeclared readout menu')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print('READOUT_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
