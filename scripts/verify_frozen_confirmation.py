"""Verify frozen training-only fits and independent confirmation evidence."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scripts.develop_forecast_readout import choose
from microservice.forecast_readout import features
from microservice.load_response import response
from microservice.workload import vector
from microservice.regime import cases,loads
from microservice.latent_probe import labels
from microservice.partial_readout import fit,score,diagnose
from microservice.workload_evaluate import stats
from collections import Counter
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/lab-confirm-v1';F=R/'reports/evidence/confirmation-readouts-v1'
def main():
    frozen=json.loads((F/'frozen.json').read_text())
    assert (E/'frozen/frozen.json').read_bytes()==(F/'frozen.json').read_bytes()
    assert (E/'frozen/readouts.npz').read_bytes()==(F/'readouts.npz').read_bytes()
    assert hashlib.sha256((F/'readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256']
    for path,h in frozen['input_hashes'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
    old=R/'reports/evidence/lab-workload-v2';audit=R/'reports/evidence/workload-forecast-audit-v1'
    train=[json.loads(l) for l in (old/'collection/train.jsonl').read_text().splitlines()]
    xs=features(np.load(old/'evaluation/train-features.npz'),np.load(audit/'train.npz'));y=labels(train);groups=np.array([r['regime'] for r in train])
    params=np.load(F/'readouts.npz');models={}
    for name,x in xs.items():
        config,counts=choose(x,y,groups);assert frozen['selection'][name]==dict(config=config,cv_correct=counts)
        m=fit(x,y,config)
        for k in ('mean','scale','bias','z','dual'):np.testing.assert_allclose(m[k],params[f'{name}_{k}'],rtol=1e-7,atol=1e-8)
        models[name]=m
    before=json.loads((E/'evaluation/before-collection.json').read_text())
    assert before['utc']>frozen['utc'] and before['checkpoint_sha256']==frozen['checkpoint_sha256']
    assert before['readouts_sha256']==frozen['readouts_sha256']
    rows=[json.loads(l) for l in (E/'evaluation/confirmation.jsonl').read_text().splitlines()]
    assert Counter((r['fault'],r['regime'],r['phase']) for r in rows)==Counter(cases('validation'))
    for r in rows:
        assert r['seed']==9002 and r['loads']==loads('validation',r['regime'],r['phase']) and r['seconds']<=35
        for frame in r['frames']:
            assert frame['successes']==frame['requests']
            assert all(m['count']==frame['requests'] and m['timestamp']>before['utc'] for m in frame['metrics'].values())
    saved=np.load(E/'evaluation/features.npz');f=np.load(E/'evaluation/forecast.npz')
    history=np.array([np.concatenate([vector(r,t) for t in range(4)]) for r in rows])
    actual=np.array([vector(r,t+1)[[0,3,4,5]] for r in rows for t in range(3)])
    current=np.array([vector(r,t)[[0,3,4,5]] for r in rows for t in range(3)])
    np.testing.assert_array_equal(f['actual'],actual);np.testing.assert_array_equal(f['persistence'],current)
    expected=features(dict(history=history,response=np.array([response(r) for r in rows]),latent=saved['latent']),
        dict(fixed=f['predicted'],current=current,actual=actual))
    for name,x in expected.items():np.testing.assert_array_equal(x,saved[name])
    np.testing.assert_array_equal(saved['labels'],labels(rows))
    target=diagnose(labels(rows));p=json.loads((E/'evaluation/predictions.json').read_text());s=json.loads((E/'evaluation/summary.json').read_text())
    assert target.tolist()==p['target']
    for name,m in models.items():assert diagnose(score(m,expected[name])).tolist()==p['methods'][name]
    for name,pred in p['methods'].items():assert stats(np.array(pred),rows)==s['methods'][name]
    for name in ('predicted','persistence'):assert float(np.mean((f[name]-actual)**2))==s['forecast_mse'][name]
    assert np.isfinite(f['predicted']).all()
    d=s['methods']['latent'];reliable=bool(d['correct']>=58 and d['families']['0']['correct']==8 and all(v['correct']>=6 for k,v in d['families'].items() if k!='0'))
    advantage=d['correct']>=max(s['methods'][k]['correct'] for k in ('response','history','response_history'))+4
    fg=s['forecast_mse']['predicted']<s['forecast_mse']['persistence']
    assert (reliable,advantage,fg)==(s['primary_reliable'],s['added_value'],s['forecast_gate'])
    assert s['combined_gate']==bool(reliable and advantage and fg)
    for name,c in s['paired'].items():
        a=np.array(p['methods']['latent'])==target;b=np.array(p['methods'][name])==target
        wins=int(sum(a&~b));losses=int(sum(~a&b));n=wins+losses
        pv=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        assert c==dict(latent_only_correct=wins,baseline_only_correct=losses,two_sided_exact_p=pv)
    assert s['checkpoint_unchanged'] and s['readouts_unchanged'] and not s['refit_after_collection']
    assert all(json.loads((E/'evaluation/audit.json').read_text()).values())
    for path,h in json.loads((E/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((E/path).read_bytes()).hexdigest()==h
    assert json.loads((E/'restored.json').read_text())['template_unchanged']
    assert 'WORKLOAD_ACTIONS_VERIFIED 192' in (E/'run.log').read_text()
    result=dict(verified=True,training_only_fits_reproduced=True,freeze_precedes_collection=True,confirmation_incidents=64,seed=9002,all_predictions_and_metrics_reproduced=True,combined_gate=s['combined_gate'],scope='Latent states and forecasts exported by the job are not regenerated locally; parameter fits, observations, timing, predictions, metrics and hashes are independently verified.')
    (E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
