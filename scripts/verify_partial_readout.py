"""Reproduce development predictions and verify nested fitting boundaries."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.partial_readout import outer_fit,folds,diagnose,assess
from microservice.efficiency import decisions
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/lab-partial-readout-v1';D=R/'reports/evidence/lab-efficiency-v1/evaluation'
def main():
    inputs=json.loads((O/'inputs.json').read_text());splits=json.loads((O/'splits.json').read_text());saved=json.loads((O/'predictions.json').read_text());summary=json.loads((O/'summary.json').read_text())
    assert set(inputs['files'])=={'development.jsonl','development-features.npz','development-scores.json'}
    for name,h in inputs['files'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h
    fs=np.load(D/'development-features.npz');target=np.array(saved['target']);families=np.array(saved['families']);y=((target[:,None]>>np.arange(3))&1)*2-1
    assert sorted(i for split in splits for i in split['test'])==list(range(80))
    for k,split in enumerate(splits):
        train=np.array(split['train']);test=np.array(split['test'])
        assert not set(train)&set(test) and set(train)|set(test)==set(range(80))
        assert all(sum(families[test]==f)==2 for f in range(8))
        assert sorted(i for inner in split['inner'] for i in inner['test'])==sorted(train)
        for inner in split['inner']:
            assert not set(inner['train'])&set(inner['test'])
            assert set(inner['train'])|set(inner['test'])==set(train)
        for name in ['latent','raw']:
            features=[fs[f'{name}_{t}'] for t in range(4)]
            run=outer_fit(features,y,families,train,test,folds(families[train],4,8302+k))
            stored=saved['methods'][name]
            np.testing.assert_allclose(run['scores'],np.array(stored['scores'])[test],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(run['fixed_scores'],np.array(stored['fixed_scores'])[test],rtol=1e-9,atol=1e-9)
            assert run['policy']==stored['folds'][k]['policy'] and run['selected']==stored['folds'][k]['selected']
            pred,checks=decisions(run['scores'],run['policy']['threshold'])
            np.testing.assert_array_equal(pred,np.array(stored['stopping_predictions'])[test]);np.testing.assert_array_equal(checks,np.array(stored['stopping_checks'])[test])
    for name in ['latent','raw']:
        stored=saved['methods'][name];tuned=summary[name]['tuned']
        assert tuned['stopping']==assess(np.array(stored['stopping_predictions']),np.array(stored['stopping_checks']),target,families)
        for method,key in [('tuned','scores'),('fixed_ridge','fixed_scores')]:
            for t in range(4):assert summary[name][method]['fixed_budgets'][str(t)]==assess(diagnose(np.array(stored[key])[:,t]),np.full(80,t),target,families)
    l=summary['latent']['tuned'];r=summary['raw']['tuned'];s=l['stopping']
    gate=bool(s['gate_passed'] and s['correct']>=l['fixed_budgets']['3']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    assert summary['development_gate_passed']==gate and summary['live_confirmation_warranted']==gate
    result=dict(verified=True,only_development_inputs=True,input_hashes_unchanged=True,outer_predictions_once_per_incident=True,nested_splits_disjoint=True,all_predictions_and_thresholds_reproduced=True,development_gate_passed=gate)
    (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
