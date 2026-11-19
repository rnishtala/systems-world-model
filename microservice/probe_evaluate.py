"""Freeze training-only readouts, then collect and score new incidents."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from microservice.agent import Model
from microservice.auxiliary_audit import audit
from microservice.probe_target import extract
from microservice.probe_collect import collect
from microservice.latent_probe import labels,digest
from microservice.partial_readout import fit,score,diagnose
from scripts.develop_forecast_readout import choose

def stats(pred,rows):
    target=diagnose(labels(rows));families=np.array([r['fault'] for r in rows])
    return dict(n=len(rows),correct=int(sum(pred==target)),families={str(f):dict(n=int(sum(families==f)),correct=int(sum((pred==target)&(families==f)))) for f in range(8)})

def main():
    out=Path('evaluation');out.mkdir(exist_ok=False);root=Path('models').resolve()
    before=digest(root/'final-91');model=Model(str(root),91,'native')
    assert model.contract=='probe-target-v1';(out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    train=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()]
    assert len(train)==64 and {r['seed'] for r in train}=={9101,9103}
    xs,aux,forecast=extract(model,train);y=labels(train);groups=np.array([r['repeat'] for r in train])
    np.savez_compressed(out/'train-features.npz',**xs,labels=y);np.savez_compressed(out/'train-forecast.npz',**forecast)
    probes={};selection={}
    for name,x in xs.items():
        config,counts=choose(x,y,groups);probes[name]=fit(x,y,config);selection[name]=dict(config=config,cv_exact_correct=counts)
    probes['shuffled_latent']=fit(xs['latent'],y[np.random.default_rng(9204).permutation(len(y))],selection['latent']['config'])
    np.savez_compressed(out/'readouts.npz',**{f'{n}_{k}':v for n,m in probes.items() for k,v in m.items() if k!='config'})
    frozen=dict(utc=time.time(),checkpoint_sha256=before,selection=selection,configs={n:m['config'] for n,m in probes.items()},
        readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest(),train_sha256=hashlib.sha256(Path('collection/train.jsonl').read_bytes()).hexdigest())
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n');print('PROBE_READOUTS_FROZEN',flush=True)
    test=collect(out/'test.jsonl');tx,ta,tf=extract(model,test)
    np.savez_compressed(out/'test-features.npz',**tx,labels=labels(test));np.savez_compressed(out/'test-forecast.npz',**tf)
    pred={n:diagnose(score(m,tx['latent' if n=='shuffled_latent' else n])) for n,m in probes.items()};pred['auxiliary']=ta
    metrics={n:stats(p,test) for n,p in pred.items()};target=diagnose(labels(test));paired={}
    for n in ('response','history','response_history'):
        wins=int(sum((pred['latent']==target)&(pred[n]!=target)));losses=int(sum((pred['latent']!=target)&(pred[n]==target)));total=wins+losses
        p=min(1.,2*sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total) if total else 1.
        paired[n]=dict(latent_only_correct=wins,control_only_correct=losses,p=p)
    forecasts={}
    for name,mask in [('all',np.ones(len(tf['actual']),bool)),('same_target',~tf['switched']),('switched_target',tf['switched'])]:
        forecasts[name]=dict(n=int(sum(mask)),**{n:float(np.mean((tf[n][mask]-tf['actual'][mask])**2)) for n in ('predicted','persistence')})
    a=metrics['latent'];reliable=bool(a['correct']>=29 and a['families']['0']['correct']==4 and all(v['correct']>=3 for v in a['families'].values()))
    matched=bool(a['correct']>=max(metrics[n]['correct'] for n in ('response','history','response_history'))-1)
    forecast_gate=forecasts['all']['predicted']<forecasts['all']['persistence']
    assert before==digest(root/'final-91') and frozen['readouts_sha256']==hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest()
    (out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),methods={n:p.tolist() for n,p in pred.items()}),indent=2)+'\n')
    result=dict(methods=metrics,forecast_mse=forecasts,paired=paired,reliable=reliable,within_one_of_strongest_control=matched,
        forecast_gate=forecast_gate,feasibility_gate=bool(reliable and matched and forecast_gate),checkpoint_unchanged=True,readouts_unchanged=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PROBE_EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
