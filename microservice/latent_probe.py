"""Frozen-state diagnostic probes; no changes to Dreamer weights or serving API."""
import argparse,hashlib,json,time,uuid
from pathlib import Path
import numpy as np
from microservice.environment import Incident
from microservice.mechanisms import FAULT_MASK


def fit_probe(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    # Fixed ridge penalty 1.0; dual solve avoids a large latent-dimension solve.
    weights=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y-bias)
    return dict(mean=mean,scale=scale,weights=weights,bias=bias)


def predict(probe,x):
    scores=(np.asarray(x)-probe['mean'])/probe['scale']@probe['weights']+probe['bias']
    return np.sum((scores>0)*np.array([1,2,4]),axis=-1)


def labels(rows):
    masks=np.array([FAULT_MASK[r['fault']] for r in rows])
    return ((masks[:,None]>>np.arange(3))&1)*2-1


def observations(row):
    sim=Incident(row['frames'],row['fault'],compositional=True)
    values=[sim.observe().tolist()]
    for action in (1,2,3):values.append(sim.step(action)[0].tolist())
    return values


def extract(model,rows):
    import jax
    raw=[];latent=[];diagnoses=[]
    for i,row in enumerate(rows):
        incident=str(uuid.UUID(int=i+1))
        # Truth is not passed into Model.act; each incident starts a fresh carry.
        for t,vector in enumerate(observations(row)):
            result=model.act(vector,t==0,0. if t==0 else -.2,incident)
        with jax.transfer_guard('allow'):
            dyn=model.agent._stack(model.carry)[1]
            state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
        raw.append(vector[2:10]);latent.append(state);diagnoses.append(result['action']-4)
        if i%20==0:print('EXTRACT',i,flush=True)
    return np.asarray(raw),np.asarray(latent),np.asarray(diagnoses)


def digest(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    from microservice.agent import Model
    before=digest(Path(a.root)/'final-20');model=Model(a.root,20,'inspect-first')
    train=list(map(json.loads,Path(a.calibration).read_text().splitlines()))
    raw,latent,_=extract(model,train);y=labels(train)
    probes={'raw':fit_probe(raw,y),'latent':fit_probe(latent,y),'shuffled_latent':fit_probe(latent,y[np.random.default_rng(7001).permutation(len(y))])}
    np.savez_compressed(out/'train-features.npz',raw=raw,latent=latent,labels=y)
    np.savez_compressed(out/'probes.npz',**{name+'_'+k:v for name,probe in probes.items() for k,v in probe.items()})
    (out/'fitted.json').write_text(json.dumps({'utc':time.time(),'calibration_sha256':hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),'checkpoint_sha256':before,'raw_dimensions':raw.shape[1],'latent_dimensions':latent.shape[1],'penalty':1.,'threshold':0.,'seed':6400,'scope':'Probes frozen before collecting fresh test incidents.'},indent=2))
    print('PROBES_FROZEN',flush=True)
    from microservice.live import configure,frame,reset,wait_ready,batch
    rng=np.random.default_rng(6400);faults=np.tile(np.arange(8),10);rng.shuffle(faults);test=[];wait_ready()
    try:
        with (out/'test.jsonl').open('x') as f:
            for i,fault in enumerate(faults):
                start=time.monotonic();n=configure(int(fault),rng,True)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup failed')
                frames=[frame(n) for _ in range(4)]
                if time.monotonic()-start>35 or any(x['successes']!=x['requests'] for x in frames):raise RuntimeError('Invalid incident')
                row=dict(episode=i,fault=int(fault),seed=6400,shifted=True,frames=frames,seconds=time.monotonic()-start)
                test.append(row);f.write(json.dumps(row)+'\n');f.flush();print('COLLECT',i,int(fault),flush=True)
    finally:reset()
    raw,latent,original=extract(model,test)
    np.savez_compressed(out/'test-features.npz',raw=raw,latent=latent,labels=labels(test))
    predictions={'dreamer_reward':original,**{name:predict(probe,raw if name=='raw' else latent) for name,probe in probes.items()}}
    target=np.array([FAULT_MASK[r['fault']] for r in test]);summary={}
    for name,pred in predictions.items():
        families={str(f):{'n':10,'correct':int(sum(pred[i]==target[i] for i,r in enumerate(test) if r['fault']==f))} for f in range(8)}
        summary[name]={'correct':int(sum(pred==target)),'n':len(test),'families':families,'gate_passed':all(v['correct']>=(10 if f=='0' else 9) for f,v in families.items())}
    after=digest(Path(a.root)/'final-20');assert before==after,'Checkpoint changed'
    result={'methods':summary,'checkpoint_unchanged':True,'scope':'One frozen model, one fresh 80-incident batch. Linear probe failure does not establish absence of information.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'predictions.json').write_text(json.dumps({'target':target.tolist(),**{k:v.tolist() for k,v in predictions.items()}},indent=2))
    print('PROBE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
