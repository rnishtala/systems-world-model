"""Separate-regime selection without refitting; gate fresh final collection."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from microservice.regime import vector,cases,loads,SEEDS
from microservice.workload import extract
from microservice.workload_collect import collect
from microservice.latent_probe import labels,digest
from microservice.partial_readout import CANDIDATES,fit,score,diagnose,assess
from microservice.mechanisms import FAULT_MASK

def select(train,y,validation,target):
    models=[fit(train,y,c) for c in CANDIDATES]
    correct=[int(sum(diagnose(score(m,validation))==target)) for m in models]
    chosen=int(np.argmax(correct))
    return models[chosen],dict(candidate_correct=correct,selected=CANDIDATES[chosen])

def stats(pred,rows):
    target=np.array([FAULT_MASK[r['fault']] for r in rows]);families=np.array([r['fault'] for r in rows])
    result=assess(pred,np.full(len(rows),4),target,families)
    result['observation_windows']=result.pop('mean_checks');return result

def eligible(results):
    d=results['latent']
    return bool(d['n']==64 and d['correct']>=58 and d['families']['0']['correct']==8
                and all(g['correct']>=6 for f,g in d['families'].items() if f!='0')
                and d['correct']>=max(results[n]['correct'] for n in ('history','latest','response'))-2)

def paired(pred,target):
    result={};latent=pred['latent']
    for name in ('history','latest','response'):
        wins=int(sum((latent==target)&(pred[name]!=target)));losses=int(sum((latent!=target)&(pred[name]==target)));n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        result[name]=dict(latent_only_correct=wins,baseline_only_correct=losses,two_sided_exact_p=p)
    return result

def predictions(probes,features,aux):
    result={name:diagnose(score(pr,features['latent' if name=='shuffled_latent' else name])) for name,pr in probes.items()}
    result['auxiliary']=aux;return result

def save_predictions(path,pred,rows):
    path.write_text(json.dumps(dict(target=[FAULT_MASK[r['fault']] for r in rows],**{k:v.tolist() for k,v in pred.items()})))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    before=digest(Path(a.root)/'final-81');model=Model(a.root,81,'native')
    assert json.loads((Path(a.root)/'training-81.json').read_text())['observation_version']=='trace-workload-v1'
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    train=[json.loads(l) for l in Path(a.data).read_text().splitlines()]
    assert len(train)==128 and all(r['seed']==8701 for r in train)
    features,_,forecast=extract(model,train)
    np.savez_compressed(out/'train-forecast.npz',**forecast);y=labels(train)
    np.savez_compressed(out/'train-features.npz',**features,labels=y)
    (out/'before-validation.json').write_text(json.dumps(dict(utc=time.time(),checkpoint_sha256=before,data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()),indent=2))
    validation=collect(out/'validation.jsonl','validation');vf,va,forecast=extract(model,validation)
    np.savez_compressed(out/'validation-forecast.npz',**forecast)
    np.savez_compressed(out/'validation-features.npz',**vf,labels=labels(validation))
    target=diagnose(labels(validation));probes={};selection={}
    for name in features:probes[name],selection[name]=select(features[name],y,vf[name],target)
    probes['shuffled_latent']=fit(features['latent'],y[np.random.default_rng(8805).permutation(len(y))],probes['latent']['config'])
    arrays={f'{name}_{k}':v for name,m in probes.items() for k,v in m.items() if k!='config'}
    np.savez_compressed(out/'readouts.npz',**arrays)
    vp=predictions(probes,vf,va);vs={name:stats(pred,validation) for name,pred in vp.items()}
    save_predictions(out/'validation-predictions.json',vp,validation)
    forecast_gate=bool(np.mean((forecast['predicted']-forecast['actual'])**2)<np.mean((forecast['persistence']-forecast['actual'])**2))
    launch=eligible(vs) and forecast_gate
    frozen=dict(utc=time.time(),checkpoint_sha256=before,data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),validation_sha256=hashlib.sha256((out/'validation.jsonl').read_bytes()).hexdigest(),selection=selection,configs={k:v['config'] for k,v in probes.items()},readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest(),validation_gate=launch,refit_after_validation=False,forecast_gate=forecast_gate)
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    result=dict(validation=dict(methods=vs,paired=paired(vp,target)),validation_gate=launch,forecast_gate=forecast_gate,final_test_collected=False)
    print('REGIME_VALIDATION_COMPLETE',json.dumps(result),flush=True)
    if launch:
        test=collect(out/'test.jsonl','test');tf,ta,forecast=extract(model,test)
        np.savez_compressed(out/'test-forecast.npz',**forecast)
        np.savez_compressed(out/'test-features.npz',**tf,labels=labels(test))
        tp=predictions(probes,tf,ta);ts={name:stats(pred,test) for name,pred in tp.items()}
        save_predictions(out/'test-predictions.json',tp,test)
        result.update(final_test_collected=True,test=dict(methods=ts,paired=paired(tp,diagnose(labels(test)))),added_value_gate=bool(ts['latent']['gate_passed'] and ts['latent']['correct']-max(ts[n]['correct'] for n in ('history','latest','response'))>=4))
    assert before==digest(Path(a.root)/'final-81')
    assert frozen['readouts_sha256']==hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest()
    result['checkpoint_unchanged']=True
    result['forecast_mse']={}
    for split in ('train','validation','test'):
        path=out/f'{split}-forecast.npz'
        if path.exists():
            f=np.load(path);result['forecast_mse'][split]={name:float(np.mean((f[name]-f['actual'])**2)) for name in ('predicted','persistence')}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('REGIME_EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
