"""Verify replay boundaries and reproduce every nested order-selection result."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.order_features import ORDERS,observed
from microservice.order_compare import KEYS,choose_order
from microservice.partial_readout import outer_fit,folds,assess,diagnose
from microservice.efficiency import decisions
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/lab-order-v1';D=R/'reports/evidence/lab-efficiency-v1/evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for name,h in json.loads((O/'frozen-hashes.json').read_text()).items():assert sha(O/name)==h
    identity=json.loads((O/'evaluation/identity.json').read_text());features=O/'evaluation/features.npz';data=D/'development.jsonl'
    assert identity['source_sha256']==sha(data) and identity['features_sha256']==sha(features)
    assert identity['checkpoint_sha256']==json.loads((D/'readouts-frozen.json').read_text())['checkpoint_sha256']
    assert identity['checkpoint_unchanged'] and identity['source_unchanged'] and all(identity['label_audit'].values())
    assert identity['verified_previous_actions']==1200 and identity['prefix_nodes_per_incident']==16
    rows=[json.loads(l) for l in data.read_text().splitlines()];assert len(rows)==80 and all(r['seed']==8201 for r in rows)
    fs=np.load(features)
    for order in ORDERS:
        key=''.join(map(str,order))
        for t in range(4):
            assert fs[f'raw_{key}_{t}'].shape==(80,14*(t+1)) and fs[f'latent_{key}_{t}'].shape==(80,640)
            np.testing.assert_array_equal(fs[f'raw_{key}_{t}'],np.array([observed(row,order[:t]).reshape(-1) for row in rows]))
            for other in ORDERS:
                if order[:t]==other[:t]:
                    second=''.join(map(str,other))
                    for name in ['raw','latent']:np.testing.assert_array_equal(fs[f'{name}_{key}_{t}'],fs[f'{name}_{second}_{t}'])
    C=O/'comparison';inputs=json.loads((C/'inputs.json').read_text());assert inputs==dict(development_sha256=sha(data),features_sha256=sha(features))
    saved=json.loads((C/'predictions.json').read_text());splits=json.loads((C/'splits.json').read_text());summary=json.loads((C/'summary.json').read_text())
    target=np.array(saved['target']);families=np.array(saved['families']);y=((target[:,None]>>np.arange(3))&1)*2-1
    from microservice.mechanisms import FAULT_MASK
    np.testing.assert_array_equal(families,[r['fault'] for r in rows]);np.testing.assert_array_equal(target,[FAULT_MASK[r['fault']] for r in rows])
    assert sorted(i for s in splits for i in s['test'])==list(range(80))
    for k,split in enumerate(splits):
        train=np.array(split['train']);test=np.array(split['test'])
        assert not set(train)&set(test) and set(train)|set(test)==set(range(80))
        inner=folds(families[train],4,8302+k)
        for j,held in enumerate(inner):
            assert split['inner'][j]['test']==train[held].tolist()
            assert split['inner'][j]['train']==train[np.setdiff1d(np.arange(64),held)].tolist()
        for name in ['latent','raw']:
            d=saved['methods'][name];runs={}
            for key in KEYS:
                run=outer_fit([fs[f'{name}_{key}_{t}'] for t in range(4)],y,families,train,test,inner);runs[key]=run
                np.testing.assert_allclose(run['scores'],np.array(d['per_order'][key]['scores'])[test],rtol=1e-9,atol=1e-9)
                pred,checks=decisions(run['scores'],run['policy']['threshold'])
                np.testing.assert_array_equal(pred,np.array(d['per_order'][key]['pred'])[test]);np.testing.assert_array_equal(checks,np.array(d['per_order'][key]['checks'])[test])
                for field in ['selected','policy','selection']:assert run[field]==d['folds'][k]['runs'][key][field]
            chosen,threshold,candidates=choose_order(runs,target[train],families[train]);detail=d['folds'][k]
            assert (chosen,threshold,candidates)==(detail['chosen'],detail['threshold'],detail['candidates'])
            assert all(d['orders'][i]==chosen for i in test)
            pred,checks=decisions(runs[chosen]['scores'],threshold)
            np.testing.assert_array_equal(pred,np.array(d['predictions'])[test]);np.testing.assert_array_equal(checks,np.array(d['checks'])[test]);np.testing.assert_allclose(runs[chosen]['scores'],np.array(d['scores'])[test],rtol=1e-9,atol=1e-9)
    for name in ['latent','raw']:
        d=saved['methods'][name];s=summary[name]
        assert s['selected']['stopping']==assess(np.array(d['predictions']),np.array(d['checks']),target,families)
        assert s['selected']['full']==assess(diagnose(np.array(d['scores'])[:,3]),np.full(80,3),target,families)
        for key,p in d['per_order'].items():
            assert s['per_order'][key]['stopping']==assess(np.array(p['pred']),np.array(p['checks']),target,families)
            for t in range(4):assert s['per_order'][key]['fixed_budgets'][str(t)]==assess(diagnose(np.array(p['scores'])[:,t]),np.full(80,t),target,families)
    l=summary['latent']['selected'];r=summary['raw']['selected'];s=l['stopping']
    gate=bool(s['gate_passed'] and s['correct']>=l['full']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    assert gate==summary['development_gate_passed']==summary['live_confirmation_warranted']
    restore=json.loads((O/'restored.json').read_text());assert restore['ready'] and restore['template_unchanged'] and restore['replicas']==json.loads((O/'before.json').read_text())['replicas']
    result=dict(verified=True,only_development_incidents=True,shared_prefixes_identical=True,raw_observation_boundaries_verified=True,actual_action_audit_passed=True,checkpoint_unchanged=True,all_nested_predictions_reproduced=True,development_gate_passed=gate,restored=True)
    (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
