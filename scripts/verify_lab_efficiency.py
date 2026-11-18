"""Recompute exported stopping results and audit independent data boundaries."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from microservice.efficiency import scores,decisions,calibrate,metrics
from microservice.latent_probe import observations
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1]
O=R/'reports/evidence/lab-efficiency-v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]
def main():
    hashes=json.loads((O/'frozen-hashes.json').read_text())
    for name,h in hashes.items():assert sha(O/name)==h,name
    root=O/'evaluation';trainpath=R/'reports/evidence/lab-fresh-v1/collection/train.jsonl'
    train=rows(trainpath);dev=rows(root/'development.jsonl');test=rows(root/'test.jsonl')
    for data,seed in [(dev,8201),(test,8202)]:
        assert len(data)==80 and Counter(r['fault'] for r in data)==Counter({f:10 for f in range(8)})
        for row in data:
            assert row['seed']==seed and row['seconds']<=35 and row['measurement_window']=='post-warmup'
            assert len(row['frames'])==4
            for t,frame in enumerate(row['frames']):
                assert frame['requests']==frame['successes']
                assert all(m['count']==(t+1)*frame['requests'] for m in frame['metrics'].values())
    prior_test=rows(R/'reports/evidence/lab-fresh-v1/evaluation/test.jsonl')
    fingerprints=lambda data:{hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() for r in data}
    groups=[fingerprints(data) for data in [train,prior_test,dev,test]]
    assert all(len(g)==80 for g in groups)
    assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
    fitted=json.loads((root/'readouts-frozen.json').read_text());frozen=json.loads((root/'frozen.json').read_text())
    assert fitted['train_sha256']==sha(trainpath)
    assert min(f['metrics']['checkout']['timestamp'] for r in dev for f in r['frames'])>fitted['utc']
    assert min(f['metrics']['checkout']['timestamp'] for r in test for f in r['frames'])>frozen['utc']
    assert fitted['checkpoint_sha256']==frozen['checkpoint_sha256']==json.loads((R/'reports/evidence/lab-fresh-v1/evaluation/fitted.json').read_text())['checkpoint_sha256']
    assert frozen['probes_sha256']==sha(root/'probes.npz')
    assert all(json.loads((root/'audit.json').read_text()).values())
    params=np.load(root/'probes.npz');summary=json.loads((root/'summary.json').read_text());predictions=json.loads((root/'predictions.json').read_text())
    for stage,data in [('development',dev),('test',test)]:
        features=np.load(root/f'{stage}-features.npz');saved=json.loads((root/f'{stage}-scores.json').read_text())
        for name in ('raw','latent'):
            recomputed=[]
            for t in range(4):
                if name=='raw':np.testing.assert_array_equal(features[f'raw_{t}'],np.array([np.asarray(observations(row)[:t+1]).reshape(-1) for row in data]))
                probe={k:params[f'{name}_{t}_{k}'] for k in ['mean','scale','weights','bias']}
                recomputed.append(scores(probe,features[f'{name}_{t}']))
            values=np.stack(recomputed,axis=1);np.testing.assert_allclose(values,saved[name],rtol=1e-9,atol=1e-9)
            if stage=='development':assert calibrate(values,data)==frozen['policies'][name]
            else:
                pred,checks=decisions(values,frozen['policies'][name]['threshold'])
                assert pred.tolist()==predictions[name]['predictions'] and checks.tolist()==predictions[name]['checks']
                for k,v in metrics(pred,checks,data).items():assert summary[name][k]==v
                for t in range(4):assert summary[name]['fixed_budgets'][str(t)]==metrics(np.sum((values[:,t]>0)*np.array([1,2,4]),axis=-1),np.full(80,t),data)
    assert predictions['target']==[FAULT_MASK[r['fault']] for r in test]
    latent=summary['latent'];raw=summary['raw']
    assert summary['success']==bool(latent['gate_passed'] and latent['correct']>=latent['fixed_budgets']['3']['correct']-1 and 3-latent['mean_checks']>=.5 and raw['mean_checks']-latent['mean_checks']>=.5)
    target=np.array(predictions['target']);lp=np.array(predictions['latent']['predictions']);rp=np.array(predictions['raw']['predictions'])
    assert summary['paired_latent_only_correct']==int(sum((lp==target)&(rp!=target)))
    assert summary['paired_raw_only_correct']==int(sum((rp==target)&(lp!=target)))
    assert summary['checkpoint_and_probes_unchanged']
    restoration=json.loads((O/'restored.json').read_text());assert restoration['ready'] and restoration['template_unchanged']
    result=dict(verified=True,no_duplicate_trajectories=True,development_after_readout_freeze=True,evaluation_after_policy_freeze=True,raw_uses_only_observed_prefixes=True,all_scores_and_predictions_recomputed=True,unchanged_seed51_checkpoint=True,restoration=restoration)
    (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
