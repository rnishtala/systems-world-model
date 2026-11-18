"""Train-only probe selection, then untouched workload-transfer evaluation."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from microservice.temporal import extract
from microservice.temporal_collect import collect
from microservice.latent_probe import labels,digest
from microservice.partial_readout import CANDIDATES,folds,fit,score,diagnose,assess
from microservice.mechanisms import FAULT_MASK


def select(x,y,families):
    splits=folds(families,5,8603);correct=[]
    for config in CANDIDATES:
        pred=np.empty(len(y),int)
        for held in splits:
            train=np.setdiff1d(np.arange(len(y)),held)
            pred[held]=diagnose(score(fit(x[train],y[train],config),x[held]))
        correct.append(int(sum(pred==diagnose(y))))
    chosen=CANDIDATES[int(np.argmax(correct))]
    return fit(x,y,chosen),dict(candidate_correct=correct,selected=chosen)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    before=digest(Path(a.root)/'final-61');model=Model(a.root,61,'native')
    assert json.loads((Path(a.root)/'training-61.json').read_text())['observation_version']=='trace-temporal-v1'
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    train=[json.loads(l) for l in Path(a.data).read_text().splitlines()]
    assert len(train)==80 and all(r['seed']==8601 and set(r['loads'])=={4,6} for r in train)
    features,_=extract(model,train);y=labels(train);families=np.array([r['fault'] for r in train])
    probes={};selected={}
    for name,x in features.items():probes[name],selected[name]=select(x,y,families)
    probes['shuffled_latent']=fit(features['latent'],y[np.random.default_rng(8605).permutation(80)],probes['latent']['config'])
    arrays={f'{name}_{k}':v for name,m in probes.items() for k,v in m.items() if k!='config'}
    np.savez_compressed(out/'readouts.npz',**arrays);np.savez_compressed(out/'train-features.npz',**features,labels=y)
    frozen=dict(utc=time.time(),checkpoint_sha256=before,data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),selection=selected,configs={k:v['config'] for k,v in probes.items()},readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest())
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2));print('TEMPORAL_READOUTS_FROZEN',flush=True)
    test=collect(out/'test.jsonl',True);features,aux=extract(model,test)
    np.savez_compressed(out/'test-features.npz',**features,labels=labels(test))
    target=np.array([FAULT_MASK[r['fault']] for r in test]);families=np.array([r['fault'] for r in test])
    predictions={name:diagnose(score(pr,features['latent' if name=='shuffled_latent' else name])) for name,pr in probes.items()};predictions['auxiliary']=aux
    results={name:assess(pred,np.full(80,4),target,families) for name,pred in predictions.items()}
    # mean_checks here is four observation windows, not active inspection decisions.
    for stats in results.values():stats['observation_windows']=stats.pop('mean_checks')
    paired={}
    for name in ('history','latest'):
        wins=int(sum((predictions['latent']==target)&(predictions[name]!=target)))
        losses=int(sum((predictions['latent']!=target)&(predictions[name]==target)))
        n=wins+losses
        pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
        paired[name]=dict(latent_only_correct=wins,baseline_only_correct=losses,two_sided_exact_p=pvalue)
    assert before==digest(Path(a.root)/'final-61')
    assert frozen['readouts_sha256']==hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest()
    result=dict(methods=results,paired=paired,checkpoint_unchanged=True,added_value_gate=bool(results['latent']['gate_passed'] and results['latent']['correct']-max(results[n]['correct'] for n in ('history','latest'))>=4),scope='One seed, known faults, trace-only view, held-out workload; no planning or early-stopping claim.')
    (out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),**{k:v.tolist() for k,v in predictions.items()})))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('TEMPORAL_EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
