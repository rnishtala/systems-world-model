"""Frozen-prefix diagnostic efficiency experiment, no serving changes."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
from microservice.latent_probe import fit_probe, labels, observations, digest
from microservice.mechanisms import FAULT_MASK


def scores(probe, x):
    return (np.asarray(x)-probe['mean'])/probe['scale']@probe['weights']+probe['bias']


def decisions(values, threshold):
    """Only the current prefix's score controls stopping; no future lookahead."""
    values=np.asarray(values)
    checks=np.full(len(values),3,dtype=int)
    if threshold is not None:
        for t in range(2,-1,-1):
            checks[np.min(np.abs(values[:,t]),axis=-1)>=threshold]=t
    chosen=values[np.arange(len(values)),checks]
    pred=np.sum((chosen>0)*np.array([1,2,4]),axis=-1)
    return pred,checks


def metrics(pred, checks, rows):
    target=np.array([FAULT_MASK[r['fault']] for r in rows])
    families={str(f):int(sum(pred[i]==target[i] for i,r in enumerate(rows) if r['fault']==f)) for f in range(8)}
    return dict(correct=int(sum(pred==target)),n=len(rows),mean_checks=float(np.mean(checks)),families=families,
                gate_passed=all(v>=(10 if f=='0' else 9) for f,v in families.items()))


def calibrate(values, rows):
    full=metrics(*decisions(values,None),rows)
    candidates=[]
    for threshold in [0.,.25,.5,1.,2.,4.,None]:
        m=metrics(*decisions(values,threshold),rows)
        candidates.append(dict(threshold=threshold,**m,eligible=m['gate_passed'] and m['correct']>=full['correct']-1))
    eligible=[c for c in candidates if c['eligible']]
    best=min(eligible,key=lambda c:(c['mean_checks'],-(float('inf') if c['threshold'] is None else c['threshold']))) if eligible else None
    return dict(threshold=best['threshold'] if best else None,development_calibration_passed=bool(best),candidates=candidates)


def extract(model,rows):
    import jax
    raw=[[] for _ in range(4)];latent=[[] for _ in range(4)]
    for i,row in enumerate(rows):
        incident=str(uuid.uuid4());history=[]
        # observations reveals only inspected groups; fault labels are not policy inputs.
        for t,vector in enumerate(observations(row)):
            history.extend(vector)
            model.act(vector,t==0,0. if t==0 else -.2,incident)
            with jax.transfer_guard('allow'):
                dyn=model.agent._stack(model.carry)[1]
                state=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
            raw[t].append(history.copy());latent[t].append(state)
        if i%20==0:print('PREFIX_EXTRACT',i,flush=True)
    return {'raw':[np.asarray(x) for x in raw],'latent':[np.asarray(x) for x in latent]}


def collect(path,seed):
    from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window
    rng=np.random.default_rng(seed);faults=np.tile(np.arange(8),10);rng.shuffle(faults);rows=[];wait_ready()
    try:
        with path.open('x') as stream:
            for i,fault in enumerate(faults):
                start=time.monotonic();n=configure(int(fault),rng,True)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup failed')
                reset_measurement_window();frames=[frame(n) for _ in range(4)]
                row=dict(episode=i,fault=int(fault),seed=seed,shifted=True,measurement_window='post-warmup',frames=frames,seconds=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if row['seconds']>35 or any(f['requests']!=f['successes'] for f in frames):raise RuntimeError('Invalid incident retained')
                if any(m['count']!=(t+1)*n for t,f in enumerate(frames) for m in f['metrics'].values()):raise RuntimeError('Contaminated window retained')
                rows.append(row);print('COLLECT',seed,i+1,flush=True)
    finally:reset()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--train',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    checkpoint=Path(a.root)/'final-51';before=digest(checkpoint);model=Model(a.root,51,'inspect-mechanisms')
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    train=[json.loads(l) for l in Path(a.train).read_text().splitlines()];features=extract(model,train)
    probes={name:[fit_probe(x,labels(train)) for x in prefixes] for name,prefixes in features.items()}
    np.savez_compressed(out/'probes.npz',**{f'{name}_{t}_{k}':v for name,ps in probes.items() for t,pr in enumerate(ps) for k,v in pr.items()})
    (out/'readouts-frozen.json').write_text(json.dumps(dict(utc=time.time(),checkpoint_sha256=before,train_sha256=hashlib.sha256(Path(a.train).read_bytes()).hexdigest()),indent=2))
    def predict_all(rows,stage):
        fs=extract(model,rows)
        np.savez_compressed(out/f'{stage}-features.npz',**{f'{name}_{t}':x for name,prefixes in fs.items() for t,x in enumerate(prefixes)})
        return {name:np.stack([scores(pr,x) for pr,x in zip(probes[name],prefixes)],axis=1) for name,prefixes in fs.items()}
    dev=collect(out/'development.jsonl',8201);dev_scores=predict_all(dev,'development')
    policies={name:calibrate(v,dev) for name,v in dev_scores.items()}
    (out/'development-scores.json').write_text(json.dumps({k:v.tolist() for k,v in dev_scores.items()}))
    frozen=dict(utc=time.time(),policies=policies,checkpoint_sha256=before,probes_sha256=hashlib.sha256((out/'probes.npz').read_bytes()).hexdigest())
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2));print('STOPPING_FROZEN',json.dumps(policies),flush=True)
    test=collect(out/'test.jsonl',8202);test_scores=predict_all(test,'test')
    (out/'test-scores.json').write_text(json.dumps({k:v.tolist() for k,v in test_scores.items()}))
    result={};predictions={}
    for name,values in test_scores.items():
        pred,checks=decisions(values,policies[name]['threshold']);result[name]=metrics(pred,checks,test)
        result[name]['fixed_budgets']={str(t):metrics(np.sum((values[:,t]>0)*np.array([1,2,4]),axis=-1),np.full(len(test),t),test) for t in range(4)}
        predictions[name]=dict(predictions=pred.tolist(),checks=checks.tolist())
    latent=result['latent'];raw=result['raw'];truth=np.array([FAULT_MASK[r['fault']] for r in test]);lp=np.array(predictions['latent']['predictions']);rp=np.array(predictions['raw']['predictions'])
    result['success']=bool(latent['gate_passed'] and latent['correct']>=latent['fixed_budgets']['3']['correct']-1 and 3-latent['mean_checks']>=.5 and raw['mean_checks']-latent['mean_checks']>=.5)
    result['paired_latent_only_correct']=int(sum((lp==truth)&(rp!=truth)));result['paired_raw_only_correct']=int(sum((rp==truth)&(lp!=truth)))
    assert digest(checkpoint)==before,'Checkpoint changed'
    assert hashlib.sha256((out/'probes.npz').read_bytes()).hexdigest()==frozen['probes_sha256'],'Probes changed'
    result['checkpoint_and_probes_unchanged']=True
    (out/'predictions.json').write_text(json.dumps(dict(target=truth.tolist(),**predictions),indent=2))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('EFFICIENCY_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
