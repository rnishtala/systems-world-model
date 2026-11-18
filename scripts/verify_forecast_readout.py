"""Reproduce nested readout selection and confirm workload-group isolation."""
import hashlib,json
import numpy as np
from scripts.develop_forecast_readout import R,OUT,SOURCE,AUDIT,features
from microservice.partial_readout import CANDIDATES,fit,score,diagnose
from microservice.latent_probe import labels
from microservice.workload_evaluate import stats

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    saved=json.loads((OUT/'predictions.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    splits=json.loads((OUT/'splits.json').read_text())
    for name,h in json.loads((OUT/'input-hashes.json').read_text()).items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
    rows=[json.loads(l) for l in (SOURCE/'collection/train.jsonl').read_text().splitlines()]
    base=np.load(SOURCE/'evaluation/train-features.npz');f=np.load(AUDIT/'train.npz')
    xs=features(base,f);y=labels(rows);target=diagnose(y);groups=np.array([r['regime'] for r in rows])
    assert saved['target']==target.tolist()
    assert sorted(i for fold in splits for i in fold['held'])==list(range(128))
    assert xs['response_forecast'].shape==(128,38)
    for name,x in xs.items():
        pred=np.empty(128,int)
        for fold,chosen in zip(splits,selection[name]):
            train=np.array(fold['train']);held=np.array(fold['held']);g=fold['regime']
            assert set(train).isdisjoint(held) and set(train)|set(held)==set(range(128))
            assert np.all(groups[held]==g) and np.all(groups[train]!=g)
            counts=[]
            for config in CANDIDATES:
                p=np.empty(len(train),int)
                for inner in sorted(set(groups[train])):
                    take=groups[train]==inner
                    fitrows=train[~take];testrows=train[take]
                    assert set(fitrows).isdisjoint(testrows) and set(fitrows).isdisjoint(held)
                    model=fit(x[fitrows],y[fitrows],config)
                    np.testing.assert_allclose(model['mean'],x[fitrows].mean(0),rtol=1e-5,atol=1e-6)
                    p[take]=diagnose(score(model,x[testrows]))
                counts.append(int(sum(p==target[train])))
            assert counts==chosen['inner_correct']
            config=CANDIDATES[int(np.argmax(counts))];assert config==chosen['config']
            pred[held]=diagnose(score(fit(x[train],y[train],config),x[held]))
        assert pred.tolist()==saved['methods'][name]
        assert stats(pred,rows)==summary['methods'][name]
    a=np.array(saved['methods']['response_forecast']);b=np.array(saved['methods']['response'])
    assert summary['paired']==dict(forecast_only_correct=int(sum((a==target)&(b!=target))),response_only_correct=int(sum((b==target)&(a!=target))))
    m=summary['methods']['response_forecast'];g=bool(m['correct']>=116 and m['families']['0']['correct']==16 and all(v['correct']>=12 for k,v in m['families'].items() if k!='0') and m['correct']>=summary['methods']['response']['correct']+4)
    assert g==summary['development_gate']
    if not g:assert not (OUT/'frozen-readouts.npz').exists()
    result=dict(verified=True,outer_groups_disjoint=True,inner_groups_disjoint=True,training_only_normalization=True,all_selections_and_predictions_reproduced=True,development_gate=g,scope='Readout-only validation; world-model training exposure remains and is not removed by these splits.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
