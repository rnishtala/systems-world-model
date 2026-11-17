"""Training-only audit of imagined check outcomes; never changes model weights."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from benchmark.acquisition import load_training,ObservationOracle,fit_shared,rank,softmax
from benchmark.supervision import TARGET,make_agent
from benchmark.acquisition_train import executed_action

def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'checkpoint').rglob('*')) if p.is_file()}

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

    before=hashes(root)
    def distribution(features):
        logits=agent.model.diagnosis(agent.model.feat2tensor(features)).astype(jnp.float32)
        probs=jax.nn.softmax(logits,-1)
        ent=-(probs*jnp.log(jnp.maximum(probs,1e-12))).sum(-1)
        return probs,ent
    @nj.pure
    def imagine(dyn):
        states=jax.tree.map(lambda v:jnp.repeat(v,160,axis=0),dyn)
        actions={'action':jnp.repeat(jnp.arange(5,dtype=jnp.int32),32)}
        _,(features,_)=agent.model.dyn.imagine(states,actions,1,False,single=True)
        probs,entropy=distribution(features)
        _,_,decoded=agent.model.dec(agent.model.dec.initial(160),features,jnp.zeros(160,bool),training=False,single=True)
        return probs.reshape(5,32,5).mean(1),entropy.reshape(5,32).mean(1),decoded['vector'].pred().reshape(5,32,dim).mean(1)
    @nj.pure
    def posterior(carry,vector):
        carry=jax.tree.map(lambda v:jnp.repeat(v,32,axis=0),carry)
        obs=dict(vector=jnp.repeat(vector[None],32,axis=0),reward=jnp.zeros(32),is_first=jnp.zeros(32,bool),is_last=jnp.zeros(32,bool),is_terminal=jnp.zeros(32,bool))
        updated,_,_=agent.model.policy(carry,obs,mode='eval')
        probs,entropy=distribution(updated[1])
        return probs.mean(0),entropy.mean()
    predict=jax.jit(lambda params,dyn,key:imagine(params,dyn,seed=key,create=False,modify=False)[1])
    observe=jax.jit(lambda params,carry,vector,key:posterior(params,carry,vector,seed=key,create=False,modify=False)[1])
    scale=x.std(0);scale[scale<1e-6]=1;mean=x.mean(0);records=[]
    for i,row in enumerate(train):
        assert row['split']=='train'
        oracle=ObservationOracle(x[i],g);carry=agent.init_policy(1)
        for budget in range(5):
            obs=oracle.observe()
            inp=dict(vector=obs[None],reward=np.zeros(1,np.float32),is_first=np.array([budget==0]),is_last=np.zeros(1,bool),is_terminal=np.zeros(1,bool),**{TARGET:np.zeros((1,5),np.float32)})
            carry,_,_=agent.policy(carry,inp,mode='eval')
            with jax.transfer_guard('allow'):
                predicted,entropies,decoded=[np.asarray(v) for v in predict(agent.params,agent._stack(carry)[1],jax.random.PRNGKey(meta['seed']*100000+i*5+budget))]
            for action in range(budget,5):
                branch=ObservationOracle(x[i],g)
                for previous in range(budget):branch.reveal(previous)
                revealed=branch.reveal(action)
                with jax.transfer_guard('allow'):
                    branchcarry=executed_action(carry,np.array([action],np.int32))
                    actual,entropy=[np.asarray(v) for v in observe(agent.params,agent._stack(branchcarry),revealed,jax.random.PRNGKey(900000+meta['seed']*10000+i*25+budget*5+action))]
                scores=(revealed-probe['mean'])/probe['scale']@probe['weights']+probe['bias'];prob=softmax(scores)
                ix=g[action];values=ix[ix<len(cols)];masks=ix[ix>=len(cols)]
                rec=dict(id=row['id'],budget=budget,action=services[action],predicted_entropy=float(entropies[action]),posterior_entropy=float(entropy),probability_brier=float(np.square(predicted[action]-actual).sum()),shared_nll=float(-np.log(max(prob[y[i]],1e-12))),shared_correct=bool(scores.argmax()==y[i]))
                for label,indices in [('values',values),('masks',masks)]:
                    rec[label+'_forecast_mse']=float(np.square((decoded[action,indices]-x[i,indices])/scale[indices]).mean())
                    rec[label+'_mean_mse']=float(np.square((mean[indices]-x[i,indices])/scale[indices]).mean())
                assert all(np.isfinite(v) for v in rec.values() if isinstance(v,float));records.append(rec)
            oracle.reveal(budget);carry=executed_action(carry,np.array([budget],np.int32))
        print('FORECAST_AUDITED',meta['seed'],i+1,flush=True)
    assert hashes(root)==before
    (out/'records.json').write_text(json.dumps(records))
    (out/'summary.json').write_text(json.dumps(dict(seed=meta['seed'],training_cases=len(train),candidate_records=len(records),seconds=time.monotonic()-start,checkpoint_hashes_unchanged=True,checkpoint_hashes=before,test_arrays_loaded=False,scope='In-sample development mechanism audit; no confirmation or promotion'),indent=2))
    print('FORECAST_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
