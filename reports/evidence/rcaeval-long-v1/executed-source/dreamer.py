"""Passive telemetry DreamerV3 experiment. No simulated remediation claims."""
import argparse,json,time,os
from pathlib import Path
import numpy as np

def sequence(path,columns):
    with np.load(path,allow_pickle=False) as d:
        t,x,names=d["time"],d["values"],d["columns"].tolist()
    result=np.zeros((20,len(columns)*2),np.float32)
    for j,col in enumerate(columns):
        if col not in names:continue
        v=x[:,names.index(col)];a=v[(t<0)&np.isfinite(v)]
        if len(a)<2:continue
        scale=np.percentile(a,75)-np.percentile(a,25)
        if scale<1e-6:scale=max(float(np.std(a)),1.)
        z=np.clip((v-np.median(a))/scale,-20,20)/20
        for k in range(20):
            b=z[(t>=-300+30*k)&(t<-300+30*(k+1))];b=b[np.isfinite(b)]
            if len(b):result[k,j]=np.mean(b);result[k,len(columns)+j]=1
    return result

def fit(x,y):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    return dict(mean=mean,scale=scale,weights=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y-bias),bias=bias)

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--train-answers",required=True);p.add_argument("--out",required=True);p.add_argument("--steps",type=int,default=8000);p.add_argument("--seed",type=int,default=42);p.add_argument("--validation-only",action="store_true");a=p.parse_args()
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/"manifest.json").read_text());train=[r for r in manifest if r["split"]=="train"]
    if a.validation_only:assert all(r["split"] in ("train","validation") for r in manifest),"Test cases prohibited"
    truth={r["id"]:r["service"] for r in json.loads(Path(a.train_answers).read_text())}
    assert set(truth)=={r["id"] for r in train},"Training labels must exclude held-out cases"
    columns=set()
    for r in train:
        with np.load(root/(r["id"]+".npz"),allow_pickle=False) as d:columns.update(d["columns"].tolist())
    columns=sorted(columns);services=sorted({c.split("_",1)[0] for c in columns})
    sequences={r["id"]:sequence(root/(r["id"]+".npz"),columns) for r in manifest}
    training=[sequences[r["id"]] for r in train]
    import sys
    sys.path.insert(0,os.environ["DREAMER_SOURCE"])
    import elements,embodied,jax
    from dreamerv3 import main as upstream
    from experiment.run import configuration
    class Env:
        def __init__(self):self.rng=np.random.default_rng(a.seed);self.done=True
        @property
        def obs_space(self):return dict(vector=elements.Space(np.float32,(2*len(columns),)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        @property
        def act_space(self):return dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,1))
        def step(self,action):
            first=bool(action["reset"]) or self.done
            if first:self.seq=training[self.rng.integers(len(training))];self.t=0
            else:self.t+=1
            self.done=self.t==19
            return dict(vector=self.seq[self.t],reward=np.float32(0),is_first=first,is_last=self.done,is_terminal=self.done)
        def close(self):pass
    c=configuration(a.seed,out/"train",a.steps).update(**{"run.train_ratio":16.,"run.save_every":300})
    upstream.make_env=lambda config,index,**kw:upstream.wrap_env(Env(),config)
    captured={}
    def build():captured["agent"]=upstream.make_agent(c);return captured["agent"]
    args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
    Path(c.logdir).mkdir(parents=True);c.save(elements.Path(c.logdir)/"config.yaml");start=time.monotonic()
    embodied.run.train(build,lambda:upstream.make_replay(c,"replay"),lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),lambda:upstream.make_logger(c),args)
    agent=captured["agent"];cp=elements.Checkpoint(elements.Path(out/"checkpoint"));cp.agent=agent;cp.save()
    print("TRAINED",time.monotonic()-start,flush=True)
    features={}
    # Reset state per incident; zero rewards and a single no-op action throughout.
    for i,r in enumerate(manifest):
        carry=agent.init_policy(1)
        for t,v in enumerate(sequences[r["id"]]):
            obs=dict(vector=v[None],reward=np.zeros(1,np.float32),is_first=np.array([t==0]),is_last=np.array([t==19]),is_terminal=np.array([t==19]))
            carry,_,_=agent.policy(carry,obs,mode="eval")
        with jax.transfer_guard("allow"):
            dyn=agent._stack(carry)[1]
            features[r["id"]]=np.concatenate([np.asarray(dyn[k]).reshape(-1) for k in ("deter","stoch")]).astype(np.float64)
        if i%25==0:print("EXTRACT",i,flush=True)
    y=np.array([[float(truth[r["id"]]==s) for s in services] for r in train])
    methods={}
    for name,values in (("dreamer_state",features),("raw_sequence",{k:v.reshape(-1).astype(np.float64) for k,v in sequences.items()})):
        assert all(np.isfinite(v).all() for v in values.values()),"Nonfinite features"
        probe=fit(np.array([values[r["id"]] for r in train]),y)
        np.savez_compressed(out/(name+"-probe.npz"),**probe)
        predictions=[]
        for r in manifest:
            scores=(values[r["id"]]-probe["mean"])/probe["scale"]@probe["weights"]+probe["bias"]
            assert np.isfinite(scores).all(),"Nonfinite scores"
            predictions.append(dict(id=r["id"],ranks=[services[j] for j in np.argsort(-scores,kind="stable")]))
        (out/(name+"-predictions.json")).write_text(json.dumps(predictions,indent=2))
        methods[name]=dict(training_cases=len(train),classes=services)
    meta=dict(methods=methods,steps=a.steps,seconds=time.monotonic()-start,seed=a.seed,columns=columns,upstream_revision="e01491fad6434b2245a3b8ca201dd7faedcc458c",scope="Passive sequence reconstruction; no learned intervention policy; single-seed pilot; ridge penalty 1 frozen before held-out scoring")
    (out/"metadata.json").write_text(json.dumps(meta,indent=2));print("DREAMER_COMPLETE",flush=True)
if __name__=="__main__":main()
