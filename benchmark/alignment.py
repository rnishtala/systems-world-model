"""Development-only alignment with cross-fitted diagnostic readouts."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
from benchmark.acquisition import load_training,ObservationOracle,fit_shared,softmax
from benchmark.acquisition_train import executed_action
from benchmark.supervision import TARGET,make_agent
from benchmark.forecast_audit import hashes

def imagined_inputs(obs,particles,indices,action,dimensions,low=None,high=None):
    result=np.repeat(obs[None],len(particles),axis=0)
    values=particles[:,indices].copy()
    if low is not None:values=np.clip(values,low[indices],high[indices])
    result[:,indices]=values;result[:,dimensions+action]=1
    return result

def scores(probe,inputs):return (inputs-probe['mean'])/probe['scale']@probe['weights']+probe['bias']
def entropy(probe,inputs):
    p=softmax(scores(probe,inputs));return float(-(p*np.log(np.maximum(p,1e-12))).sum(-1).mean())
def main():
    p=argparse.ArgumentParser()
    for name in ('data','train-answers','root','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();start=time.monotonic();out=Path(a.out);out.mkdir(exist_ok=False)
    data=Path(a.data);root=Path(a.root);meta=json.loads((root/'metadata.json').read_text())
    rows,train,services,cols,g,x,y=load_training(data,Path(a.train_answers));assert cols==meta['columns'] and services==meta['classes']
    probe=fit_shared(x,y,g);np.savez_compressed(out/'shared-classifier.npz',**probe)
    classifier_hash=hashlib.sha256(b''.join(v.tobytes() for k,v in sorted(probe.items()))).hexdigest()
    sys.path.insert(0,os.environ['DREAMER_SOURCE'])
    import elements,jax,jax.numpy as jnp,ninjax as nj
    from dreamerv3 import main as up
    dim=meta['dimension']
    class Shape:
        obs_space=dict(vector=elements.Space(np.float32,(dim,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool),**{TARGET:elements.Space(np.float32,(5,),0,1)})
        act_space=dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,5))
        def close(self):pass
    c=elements.Config.load(elements.Path(root/'train/config.yaml')).update(logdir=str(out/'inference'))
    up.make_env=lambda config,index,**kw:up.wrap_env(Shape(),config)
    agent=make_agent(c,5);cp=elements.Checkpoint(elements.Path(root/'checkpoint'));cp.agent=agent;cp.load(keys=['agent'])


    before=hashes(root);fold=np.zeros(len(train),int)
    for label in range(5):
        indices=sorted(np.flatnonzero(y==label),key=lambda i:train[i]['id'])
        for j,i in enumerate(indices):fold[i]=j%2
    probes={f:fit_shared(x[fold!=f],y[fold!=f],g) for f in (0,1)}
    bounds={f:(x[fold!=f].min(0),x[fold!=f].max(0)) for f in (0,1)}
    for low,high in bounds.values():low[len(cols):]=0;high[len(cols):]=1
    @nj.pure
    def imagine(dyn):
        states=jax.tree.map(lambda v:jnp.repeat(v,160,axis=0),dyn)
        actions={'action':jnp.repeat(jnp.arange(5,dtype=jnp.int32),32)}
        _,(features,_)=agent.model.dyn.imagine(states,actions,1,False,single=True)
        p=jax.nn.softmax(agent.model.diagnosis(agent.model.feat2tensor(features)).astype(jnp.float32),-1)
        ent=-(p*jnp.log(jnp.maximum(p,1e-12))).sum(-1).reshape(5,32).mean(1)
        _,_,decoded=agent.model.dec(agent.model.dec.initial(160),features,jnp.zeros(160,bool),training=False,single=True)
        return ent,decoded['vector'].pred().reshape(5,32,dim)
    predict=jax.jit(lambda params,dyn,key:imagine(params,dyn,seed=key,create=False,modify=False)[1])
    records=[];errors=[];mean=x.mean(0);std=x.std(0);scale=std.copy();scale[scale<1e-6]=1
    for i,row in enumerate(train):
        probe=probes[fold[i]];low,high=bounds[fold[i]]
        for method in ('auxiliary','matched','matched_clipped','fixed'):
            oracle=ObservationOracle(x[i],g);carry=agent.init_policy(1);order=[]
            for budget in range(5):
                obs=oracle.observe()
                inp=dict(vector=obs[None],reward=np.zeros(1,np.float32),is_first=np.array([budget==0]),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool),**{TARGET:np.zeros((1,5),np.float32)})
                carry,_,_=agent.policy(carry,inp,mode='eval')
                with jax.transfer_guard('allow'):
                    auxiliary,decoded=[np.asarray(v) for v in predict(agent.params,agent._stack(carry)[1],jax.random.PRNGKey(meta['seed']*100000+i*5+budget))]
                available=np.flatnonzero(~oracle.seen);values=np.full(5,np.inf)
                for action in available:
                    if method=='auxiliary':values[action]=auxiliary[action]
                    elif method!='fixed':
                        hypothetical=imagined_inputs(obs,decoded[action],g[action],action,len(x[i]),low if method=='matched_clipped' else None,high if method=='matched_clipped' else None)
                        values[action]=entropy(probe,hypothetical)
                    if method=='fixed':
                        predicted=decoded[action].mean(0)
                        for j in g[action]:
                            errors.append(dict(feature=int(j),squared_error=float((predicted[j]-x[i,j])**2),baseline_error=float((mean[j]-x[i,j])**2),range_violation=bool(predicted[j]<x[:,j].min() or predicted[j]>x[:,j].max())))
                action=int(available[0]) if method=='fixed' else int(values.argmin())
                revealed=oracle.reveal(action);order.append(services[action]);prediction=int(scores(probe,revealed).argmax())
                records.append(dict(id=row['id'],fold=int(fold[i]),method=method,budget=budget+1,checks=order.copy(),correct=bool(prediction==y[i]),prediction=services[prediction],truth=services[y[i]],candidate_scores=[None if not np.isfinite(v) else float(v) for v in values]))
                carry=executed_action(carry,np.array([action],np.int32))
        print('ALIGNMENT',meta['seed'],i+1,flush=True)
    feature_stats=[]
    for j in range(len(x[0])):
        rowsj=[e for e in errors if e['feature']==j]
        mse=float(np.mean([e['squared_error'] for e in rowsj]));baseline=float(np.mean([e['baseline_error'] for e in rowsj]))
        feature_stats.append(dict(feature=cols[j%len(cols)],kind='value' if j<len(cols) else 'mask',n=len(rowsj),std=float(std[j]),raw_mse=mse,standardized_mse=mse/float(scale[j]**2),baseline_raw_mse=baseline,range_violation_rate=float(np.mean([e['range_violation'] for e in rowsj]))))
    assert hashes(root)==before
    result={method:[sum(v['correct'] for v in records if v['method']==method and v['budget']==b) for b in range(1,6)] for method in ('auxiliary','matched','matched_clipped','fixed')}
    (out/'records.json').write_text(json.dumps(records));(out/'features.json').write_text(json.dumps(feature_stats,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(seed=meta['seed'],curves=result,n=len(train),fold_ids={str(f):[row['id'] for i,row in enumerate(train) if fold[i]==f] for f in (0,1)},seconds=time.monotonic()-start,checkpoint_hashes=before,checkpoint_hashes_unchanged=True,test_arrays_loaded=False,world_model_training_exposure=True),indent=2))
if __name__=='__main__':main()
