"""Audit frozen policies, fresh measurements and reproduced confirmation scores."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from microservice.order_features import ORDERS,observed
from microservice.order_compare import KEYS,choose_order
from microservice.partial_readout import outer_fit,folds,fit,score,assess,diagnose
from microservice.efficiency import decisions
from microservice.order_confirm import load_readout
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/lab-order-confirm-v1';D=R/'reports/evidence/lab-efficiency-v1/evaluation/development.jsonl';F=R/'reports/evidence/lab-order-v1/evaluation/features.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for name,h in json.loads((O/'frozen-hashes.json').read_text()).items():assert sha(O/name)==h
    policy=json.loads((O/'frozen/policy.json').read_text());params=np.load(O/'frozen/readouts.npz');before=json.loads((O/'evaluation/before.json').read_text())
    assert policy['live_eligible'] and policy['development_sha256']==sha(D) and policy['features_sha256']==sha(F)
    assert policy['readouts_sha256']==before['readouts_sha256']==sha(O/'frozen/readouts.npz')
    assert before['policy_sha256']==sha(O/'frozen/policy.json')
    assert before['checkpoint_sha256']==policy['checkpoint_sha256']==json.loads((F.parent/'identity.json').read_text())['checkpoint_sha256']
    assert all(before['label_audit'].values())
    dev=[json.loads(l) for l in D.read_text().splitlines()];families=np.array([r['fault'] for r in dev]);target=np.array([FAULT_MASK[f] for f in families]);y=((target[:,None]>>np.arange(3))&1)*2-1;fs=np.load(F);inner=folds(families,5,8400)
    # Reproduce final fit and policy selection solely from the development inputs.
    for name in ['latent','raw']:
        runs={key:outer_fit([fs[f'{name}_{key}_{t}'] for t in range(4)],y,families,np.arange(80),np.array([],int),inner) for key in KEYS}
        chosen,threshold,candidates=choose_order(runs,target,families);p=policy['policies'][name]
        assert (chosen,threshold,candidates)==(p['order'],p['threshold'],p['candidates'])
        assert runs[chosen]['selected']==p['readouts'] and runs[chosen]['selection']==p['selection']
        for t,config in enumerate(p['readouts']):
            model=fit(fs[f'{name}_{chosen}_{t}'],y,config)
            for k in ['mean','scale','bias','z','dual']:np.testing.assert_allclose(model[k],params[f'{name}_{t}_{k}'],rtol=1e-9,atol=1e-9)
    rows=[json.loads(l) for l in (O/'evaluation/test.jsonl').read_text().splitlines()]
    assert len(rows)==80 and Counter(r['fault'] for r in rows)==Counter({f:10 for f in range(8)})
    for row in rows:
        assert row['seed']==8401 and row['shifted'] and row['measurement_window']=='post-warmup' and row['seconds']<=35
        assert len(row['frames'])==4
        for t,frame in enumerate(row['frames']):
            assert frame['successes']==frame['requests']
            assert all(m['count']==(t+1)*frame['requests'] for m in frame['metrics'].values())
    hashes=lambda data:{hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() for r in data}
    assert len(hashes(rows))==80 and not hashes(rows)&hashes(dev)
    assert min(f['metrics']['checkout']['timestamp'] for r in rows for f in r['frames'])>max(before['utc'],policy['utc'])
    xs=np.load(O/'evaluation/features.npz')
    for order in ORDERS:
        key=''.join(map(str,order))
        for t in range(4):
            np.testing.assert_array_equal(xs[f'raw_{key}_{t}'],np.array([observed(row,order[:t]).reshape(-1) for row in rows]))
            for other in ORDERS:
                if order[:t]==other[:t]:np.testing.assert_array_equal(xs[f'latent_{key}_{t}'],xs[f'latent_{"".join(map(str,other))}_{t}'])
    saved=json.loads((O/'evaluation/predictions.json').read_text());summary=json.loads((O/'evaluation/summary.json').read_text());families=np.array([r['fault'] for r in rows]);target=np.array([FAULT_MASK[f] for f in families])
    np.testing.assert_array_equal(target,saved['target']);np.testing.assert_array_equal(families,saved['families'])
    for name in ['latent','raw']:
        p=policy['policies'][name];values=np.stack([score(load_readout(params,policy,name,t),xs[f'{name}_{p["order"]}_{t}']) for t in range(4)],axis=1)
        d=saved['methods'][name];np.testing.assert_allclose(values,d['scores'],rtol=1e-9,atol=1e-9)
        pred,checks=decisions(values,p['threshold']);np.testing.assert_array_equal(pred,d['predictions']);np.testing.assert_array_equal(checks,d['checks'])
        assert summary[name]['order']==p['order'] and summary[name]['threshold']==p['threshold']
        assert summary[name]['stopping']==assess(pred,checks,target,families)
        for t in range(4):assert summary[name]['fixed_budgets'][str(t)]==assess(diagnose(values[:,t]),np.full(80,t),target,families)
        assert summary[name]['full']==summary[name]['fixed_budgets']['3']
    l=summary['latent'];r=summary['raw'];s=l['stopping']
    gate=bool(s['gate_passed'] and s['correct']>=l['full']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    assert gate==summary['confirmation_passed'] and summary['checkpoint_and_policy_unchanged'] and summary['verified_previous_actions']==1200
    lp=np.array(saved['methods']['latent']['predictions']);rp=np.array(saved['methods']['raw']['predictions'])
    assert summary['paired_latent_only_correct']==int(sum((lp==target)&(rp!=target))) and summary['paired_raw_only_correct']==int(sum((rp==target)&(lp!=target)))
    restore=json.loads((O/'restored.json').read_text());assert restore['ready'] and restore['template_unchanged'] and restore['replicas']==json.loads((O/'before.json').read_text())['replicas']
    result=dict(verified=True,final_policy_reproduced_from_development=True,all_frozen_parameters_reproduced=True,collection_after_freeze=True,all_measurement_windows_clean=True,no_development_overlap=True,observed_prefixes_only=True,all_predictions_reproduced=True,checkpoint_and_policy_unchanged=True,restored=True,confirmation_passed=gate)
    (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
