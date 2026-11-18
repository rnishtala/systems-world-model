"""Reproduce saved folds, inner candidate scores, selections and predictions."""
import json
import hashlib
import numpy as np
from scripts.develop_mechanism_readout import R, OUT, SOURCE, AUDIT, features
from microservice.partial_readout import CANDIDATES, fit, score, diagnose
from microservice.latent_probe import labels
from microservice.workload_evaluate import stats

def main():
    hashes=json.loads((OUT/'input-hashes.json').read_text())
    for name,digest in hashes.items(): assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
    rows=[json.loads(l) for l in (SOURCE/'collection/train.jsonl').read_text().splitlines()]
    xs=features(np.load(SOURCE/'evaluation/train-features.npz'),np.load(AUDIT/'train.npz')); y=labels(rows)
    splits=json.loads((OUT/'splits.json').read_text()); selection=json.loads((OUT/'selection.json').read_text())
    saved=json.loads((OUT/'predictions.json').read_text()); summary=json.loads((OUT/'summary.json').read_text())
    np.testing.assert_array_equal(saved['target'],diagnose(y))
    seen=[]
    for fold in splits:
        train=np.array(fold['train']); held=np.array(fold['held']); seen.extend(held.tolist())
        assert not set(train)&set(held) and len(train)==96 and len(held)==32
        assert {rows[i]['regime'] for i in held}=={fold['group']}
        for name,choices in selection.items():
            choice=next(c for c in choices if c['group']==fold['group'])
            trial_list=choice.get('trials',[dict(feature=name,config=c) for c in CANDIDATES])
            counts=[]
            for trial in trial_list:
                x=xs[trial['feature']]; values=np.empty((128,3))
                for inner in fold['inner']:
                    it=np.array(inner['train']); ih=np.array(inner['held'])
                    assert not set(it)&set(ih) and set(it)|set(ih)==set(train)
                    assert not (set(it)|set(ih))&set(held)
                    values[ih]=score(fit(x[it],y[it],trial['config']),x[ih])
                bits=np.sum((values[train]>0)==(y[train]>0),axis=0).tolist()
                if 'trials' in choice: assert bits==trial['bit_correct']
                counts.append(bits if 'trials' in choice else int(sum(diagnose(values[train])==diagnose(y[train]))))
            if 'trials' in choice:
                chosen=np.argmax(np.array(counts),axis=0).tolist();assert chosen==choice['chosen']
                selected=[trial_list[i] for i in chosen]; assert selected==choice['selected']
                values=np.column_stack([score(fit(xs[c['feature']][train],y[train,b:b+1],c['config']),xs[c['feature']][held])[:,0] for b,c in enumerate(selected)])
            else:
                assert counts==choice['inner_correct']
                assert CANDIDATES[int(np.argmax(counts))]==choice['config']
                values=score(fit(xs[name][train],y[train],choice['config']),xs[name][held])
            np.testing.assert_array_equal(diagnose(values),np.array(saved['methods'][name])[held])
    assert sorted(seen)==list(range(128))
    previous=json.loads((R/'reports/evidence/forecast-readout-development-v1/predictions.json').read_text())
    for name,p in saved['methods'].items():
        metric=stats(np.array(p),rows)
        for k,v in metric.items(): assert summary['methods'][name][k]==v
    for name in ('latent','response'): assert saved['methods'][name]==previous['methods'][name]
    assert summary['development_gate'] is False
    print('VERIFIED: hashes, disjoint nested folds, all inner scores and selections, predictions, metrics, unchanged baselines.')
if __name__=='__main__': main()
