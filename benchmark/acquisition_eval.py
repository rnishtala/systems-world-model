"""Frozen imagined-state acquisition; all policies share one training-only classifier."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from benchmark.acquisition import load_training,ObservationOracle,aggregate,fit_shared,rank,retrieval_action,random_order
from benchmark.supervision import TARGET,make_agent
from benchmark.acquisition_train import executed_action


def main():
    p=argparse.ArgumentParser()
    for name in ('data','train-answers','root','out'):p.add_argument('--'+name,required=True)
    p.add_argument("--smoke",action="store_true");a=p.parse_args();start=time.monotonic();out=Path(a.out);out.mkdir(exist_ok=False)
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
    @nj.pure
    def expected_entropy(dyn):
        states=jax.tree.map(lambda v:jnp.repeat(v,5*32,axis=0),dyn)
        actions={'action':jnp.repeat(jnp.arange(5,dtype=jnp.int32),32)}
        _,(features,_)=agent.model.dyn.imagine(states,actions,1,False,single=True)
        logits=agent.model.diagnosis(agent.model.feat2tensor(features)).astype(jnp.float32)
        logp=jax.nn.log_softmax(logits,-1);entropy=-(jnp.exp(logp)*logp).sum(-1)
        return entropy.reshape(5,32).mean(1)
    predict=jax.jit(lambda params,dyn,key:expected_entropy(params,dyn,seed=key,create=False,modify=False)[1])
    test=train[:1] if a.smoke else [r for r in rows if r['split']=='test'];result=[]
    for i,row in enumerate(test):
        full=aggregate(data/(row['id']+'.npz'),cols)
        for method in ('dreamer','fixed','retrieval','random'):
            for replicate in range(16 if method=='random' else 1):
                oracle=ObservationOracle(full,g);order=[];trace=[];carry=agent.init_policy(1) if method=='dreamer' else None
                permutation=random_order(row['id'],replicate) if method=='random' else list(range(5))
                for budget in range(1,6):
                    obs=oracle.observe();assert np.isfinite(obs).all()
                    # Before querying, all as-yet hidden metric coordinates must be zero.
                    for service,ix in enumerate(g):
                        if not oracle.seen[service]:assert not np.any(obs[ix])
                    if method=='dreamer':
                        inp=dict(vector=obs[None],reward=np.zeros(1,np.float32),is_first=np.array([budget==1]),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool),**{TARGET:np.zeros((1,5),np.float32)})
                        carry,_,_=agent.policy(carry,inp,mode='eval')
                        with jax.transfer_guard('allow'):
                            scores=np.asarray(predict(agent.params,agent._stack(carry)[1],jax.random.PRNGKey(meta['seed']*10000+i*5+budget))).copy()
                            assert np.isfinite(scores).all();scores[oracle.seen]=np.inf;action=int(scores.argmin())
                            carry=executed_action(carry,np.array([action],np.int32))
                        trace.append([None if not np.isfinite(v) else float(v) for v in scores])
                    elif method=='retrieval':action=retrieval_action(obs,oracle.seen,x,y,g)
                    else:action=permutation[budget-1]
                    revealed=oracle.reveal(action);order.append(services[action]);ranks=[services[j] for j in rank(probe,revealed)]
                    result.append(dict(id=row['id'],method=method,replicate=replicate,budget=budget,checks=order.copy(),ranks=ranks,predicted_entropies=trace[-1] if method=='dreamer' else None))
                assert len(set(order))==5
        if i%10==0:print('ACQUISITION_EVALUATED',i,flush=True)
    (out/'predictions.json').write_text(json.dumps(result))
    (out/'summary.json').write_text(json.dumps(dict(smoke_training_case_only=a.smoke,seed=meta['seed'],seconds=time.monotonic()-start,test_cases=len(test),classifier_sha256=classifier_hash,particles=32,random_replicates=16,test_labels_loaded=False,hidden_coordinates_zero_before_every_check=True,all_checks_distinct=True,policy_receives_only_revealed_observations=True),indent=2));print('ACQUISITION_EVAL_COMPLETE',flush=True)
if __name__=='__main__':main()
