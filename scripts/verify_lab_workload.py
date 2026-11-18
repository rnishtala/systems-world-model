"""Reproduce regime-separated selection and predictions from exported evidence."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from microservice.regime import cases,loads
from microservice.workload import vector
from microservice.load_response import response
SEEDS={"train":8701,"validation":8802,"test":8803}
from microservice.workload_evaluate import select,stats,eligible,paired
from microservice.partial_readout import fit,score,diagnose
from microservice.latent_probe import labels
from scripts.verify_lab_temporal import wilson
R=Path(__file__).resolve().parents[1]/'reports/evidence/lab-workload-v2'

def main():
    frozen=json.loads((R/'evaluation/frozen.json').read_text())
    summary=json.loads((R/'evaluation/summary.json').read_text())
    before=json.loads((R/'evaluation/before-validation.json').read_text())
    metadata=json.loads((R/'models/training-81.json').read_text())
    assert metadata['observation_version']=='trace-workload-v1' and metadata['steps']==100000 and not metadata['resumed']
    assert frozen['refit_after_validation'] is False and before['checkpoint_sha256']==frozen['checkpoint_sha256']
    params=np.load(R/'evaluation/readouts.npz')
    assert hashlib.sha256((R/'evaluation/readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256']
    splits=['train','validation']+(['test'] if summary['final_test_collected'] else [])
    rows={};features={}
    for split in splits:
        path=R/('collection/train.jsonl' if split=='train' else f'evaluation/{split}.jsonl')
        rows[split]=[json.loads(l) for l in path.read_text().splitlines()]
        assert Counter((r['fault'],r['regime'],r['phase']) for r in rows[split])==Counter(cases(split))
        for r in rows[split]:
            assert r['seed']==SEEDS[split] and r['split']==split and r['loads']==loads(split,r['regime'],r['phase'])
            assert r['seconds']<=35 and len(r['frames'])==4
            for t,frame in enumerate(r['frames']):
                assert frame['requests']==frame['successes']==r['loads'][t]
                assert all(m['count']==frame['requests'] for m in frame['metrics'].values())
                if split!='train':assert all(m['timestamp']>(before if split=='validation' else frozen)['utc'] for m in frame['metrics'].values())
        features[split]=np.load(R/f'evaluation/{split}-features.npz')
        expected=np.array([np.array([vector(r,t) for t in range(4)]).reshape(-1) for r in rows[split]])
        np.testing.assert_array_equal(features[split]['history'],expected)
        np.testing.assert_array_equal(features[split]['latest'],expected[:,-14:])
        np.testing.assert_array_equal(features[split]['response'],np.array([response(r) for r in rows[split]]))
        f=np.load(R/f'evaluation/{split}-forecast.npz')
        actual=np.array([vector(r,t)[[0,3,4,5]] for r in rows[split] for t in range(1,4)])
        persistence=np.array([vector(r,t)[[0,3,4,5]] for r in rows[split] for t in range(3)])
        np.testing.assert_array_equal(f['actual'],actual);np.testing.assert_array_equal(f['persistence'],persistence)
        for key in ('predicted','persistence'):
            assert float(np.mean((f[key]-f['actual'])**2))==summary['forecast_mse'][split][key]
        np.testing.assert_array_equal(features[split]['labels'],labels(rows[split]))
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        if split=='train':assert h==frozen['data_sha256']==before['data_sha256']==metadata['calibration_sha256']
        if split=='validation':assert h==frozen['validation_sha256']
    if not summary['final_test_collected']:
        assert not any((R/'evaluation'/name).exists() for name in ('test.jsonl','test-features.npz','test-predictions.json'))
    y=labels(rows['train']);target=diagnose(labels(rows['validation']));models={}
    for name in ('history','latest','response','latent','shuffled_latent'):
        key='latent' if name=='shuffled_latent' else name
        if name=='shuffled_latent':model=fit(features['train'][key],y[np.random.default_rng(8805).permutation(len(y))],frozen['configs'][name])
        else:
            model,selected=select(features['train'][key],y,features['validation'][key],target)
            assert selected==frozen['selection'][name]
        assert model['config']==frozen['configs'][name]
        for k in ('mean','scale','bias','z','dual'):np.testing.assert_allclose(model[k],params[f'{name}_{k}'],rtol=1e-7,atol=1e-8)
        models[name]=model
    intervals={}
    for split in splits[1:]:
        saved=json.loads((R/f'evaluation/{split}-predictions.json').read_text())
        target=diagnose(labels(rows[split]));assert saved['target']==target.tolist()
        predictions={}
        for name,model in models.items():
            key='latent' if name=='shuffled_latent' else name
            pred=diagnose(score(model,features[split][key]));assert pred.tolist()==saved[name]
            predictions[name]=pred
        predictions['auxiliary']=np.array(saved['auxiliary'])
        for name,pred in predictions.items():assert stats(pred,rows[split])==summary[split]['methods'][name]
        assert paired(predictions,target)==summary[split]['paired']
        intervals[split]={name:wilson(s['correct'],s['n']) for name,s in summary[split]['methods'].items()}
    forecast_gate=summary['forecast_mse']['validation']['predicted']<summary['forecast_mse']['validation']['persistence']
    assert forecast_gate==summary['forecast_gate']==frozen['forecast_gate']
    gate=eligible(summary['validation']['methods']) and forecast_gate
    assert gate==summary['validation_gate']==frozen['validation_gate']==summary['final_test_collected']
    if gate:
        m=summary['test']['methods'];assert summary['added_value_gate']==bool(m['latent']['gate_passed'] and m['latent']['correct']-max(m[n]['correct'] for n in ('history','latest','response'))>=4)
    assert summary['checkpoint_unchanged']
    assert all(json.loads((R/'evaluation/audit.json').read_text()).values())
    assert json.loads((R/'restored.json').read_text())['template_unchanged']
    manifest=json.loads((R/'frozen-hashes.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
    waits=sum(len(rows[s])*3 for s in splits)+6  # Separate two-incident smoke extraction.
    log=(R/'run.log').read_text()
    assert sum(int(l.split()[-1]) for l in log.splitlines() if l.startswith('WORKLOAD_ACTIONS_VERIFIED'))==waits
    assert 'WORKLOAD_PREFLIGHT_PASSED' in log and 'WORKLOAD_SMOKE_PASSED' in log
    result=dict(verified=True,final_test_collected=gate,verified_workload_actions=waits,accuracy_intervals_95=intervals,
                scope='Independent refit from training only, validation selection reproduction, observation and split checks, checkpoint/label and continuous-action alignment audits recorded by job; latent/auxiliary features not recomputed locally; validation intervals are descriptive after selection.')
    (R/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
