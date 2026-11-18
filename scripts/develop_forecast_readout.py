"""Training-only nested regime CV for a fixed forecast-informed readout."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.partial_readout import CANDIDATES,fit,score,diagnose
from microservice.latent_probe import labels
from microservice.workload_evaluate import stats
R=Path(__file__).resolve().parents[1]
SOURCE=R/'reports/evidence/lab-workload-v2'
AUDIT=R/'reports/evidence/workload-forecast-audit-v1'
OUT=R/'reports/evidence/forecast-readout-development-v1'

def features(base,forecast):
    n=len(base['response'])
    change=(forecast['fixed']-forecast['current']).reshape(n,12)
    residual=(forecast['actual']-forecast['fixed']).reshape(n,12)
    return dict(response=base['response'],history=base['history'],latent=base['latent'],
        response_history=np.concatenate([base['response'],base['history']],1),
        response_latent=np.concatenate([base['response'],base['latent']],1),
        response_forecast=np.concatenate([base['response'],change,residual],1))

def choose(x,y,groups):
    target=diagnose(y);counts=[]
    for config in CANDIDATES:
        pred=np.empty(len(y),int)
        for group in sorted(set(groups)):
            held=groups==group;train=~held
            pred[held]=diagnose(score(fit(x[train],y[train],config),x[held]))
        counts.append(int(sum(pred==target)))
    return CANDIDATES[int(np.argmax(counts))],counts

def main():
    OUT.mkdir(exist_ok=False)
    paths=[SOURCE/'collection/train.jsonl',SOURCE/'evaluation/train-features.npz',AUDIT/'train.npz']
    hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows=[json.loads(l) for l in paths[0].read_text().splitlines()]
    assert len(rows)==128 and all(r['seed']==8701 for r in rows)
    base=np.load(paths[1]);forecast=np.load(paths[2]);xs=features(base,forecast)
    y=labels(rows);groups=np.array([r['regime'] for r in rows]);target=diagnose(y)
    np.testing.assert_array_equal(base['labels'],y)
    metrics={};predictions={};selections={};splits=[]
    for g in sorted(set(groups)):
        train=np.where(groups!=g)[0];held=np.where(groups==g)[0]
        assert len(held)==32
        splits.append(dict(regime=int(g),train=train.tolist(),held=held.tolist()))
    for name,x in xs.items():
        pred=np.empty(128,int);selection=[]
        for fold in splits:
            train=np.array(fold['train']);held=np.array(fold['held'])
            config,counts=choose(x[train],y[train],groups[train])
            pred[held]=diagnose(score(fit(x[train],y[train],config),x[held]))
            selection.append(dict(regime=fold['regime'],config=config,inner_correct=counts))
        metrics[name]=stats(pred,rows);predictions[name]=pred.tolist();selections[name]=selection
        print(name,metrics[name]['correct'],flush=True)
    primary=metrics['response_forecast'];baseline=metrics['response']
    gate=bool(primary['correct']>=116 and primary['families']['0']['correct']==16
        and all(v['correct']>=12 for k,v in primary['families'].items() if k!='0')
        and primary['correct']>=baseline['correct']+4)
    a=np.array(predictions['response_forecast']);b=np.array(predictions['response'])
    paired=dict(forecast_only_correct=int(sum((a==target)&(b!=target))),
        response_only_correct=int(sum((b==target)&(a!=target))))
    result=dict(scope='Readout development only; the frozen world model trained on all 128 incidents.',
        methods=metrics,development_gate=gate,paired=paired)
    for name,data in [('summary',result),('predictions',dict(target=target.tolist(),methods=predictions)),
                      ('selection',selections),('splits',splits),('input-hashes',hashes)]:
        (OUT/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    if gate:
        models={};configs={}
        for name,x in xs.items():
            config,counts=choose(x,y,groups);models[name]=fit(x,y,config)
            configs[name]=dict(config=config,cv_correct=counts)
        np.savez_compressed(OUT/'frozen-readouts.npz',
            **{f'{n}_{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
        (OUT/'frozen-configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    print('DEVELOPMENT_GATE',gate,flush=True)
if __name__=='__main__':main()
