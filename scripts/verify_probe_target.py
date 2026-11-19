"""Independent local verification of exported probe-target experiment evidence."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from microservice.probe_target import frames,vector
from microservice.latent_probe import labels
from microservice.partial_readout import fit,score,diagnose
from scripts.develop_forecast_readout import choose
from scripts.evaluate_ledger_probe import features as raw_features
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-target-model-v1'

def main():
    e=O/'evaluation';frozen=json.loads((e/'frozen.json').read_text());summary=json.loads((e/'summary.json').read_text())
    runtime=json.loads((O/'runtime-source-hashes.json').read_text())
    earlier=json.loads((R/'reports/evidence/workload-forecast-audit-v1/upstream-source-hashes.json').read_text())
    for path,key in [('dreamerv3/agent.py','agent'),('dreamerv3/rssm.py','rssm'),('embodied/jax/heads.py','heads')]:assert runtime[path]==earlier[key]
    config=(O/'runtime-config.yaml').read_text()
    assert 'seed: 91' in config and 'steps: 100000.0' in config and 'train_ratio: 16.0' in config
    for n,h in json.loads((O/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h
    for n,h in json.loads((O/'collection/source-hashes.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    original=[]
    for name,h in json.loads((O/'training-sources.json').read_text()).items():
        p=R/f'reports/evidence/ledger-probe-v1/{name}.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()==h
        original += [json.loads(l) for l in p.read_text().splitlines()]
    train=[json.loads(l) for l in (O/'collection/train.jsonl').read_text().splitlines()];test=[json.loads(l) for l in (e/'test.jsonl').read_text().splitlines()]
    assert train==original and len(train)==64 and len(test)==32
    assert {r['seed'] for r in train}=={9101,9103} and {r['seed'] for r in test}=={9202}
    assert Counter((r['fault'],r['repeat']) for r in test)==Counter((f,j) for f in range(8) for j in range(4))
    for row in test:
        assert row['utc']>frozen['utc'] and row['seconds']<35
        for f in frames(row):
            assert f['requests']==f['successes'] and f['requests'] in (3,9)
            for s,m in f['metrics'].items():
                assert m['count']==(f['requests'] if f['target']=='checkout' or s=='ledger' else 0)
                assert m['error_rate']==0 and set(m)=={'count','latency_p95','self_p95','downstream_p95','error_rate'}
    assert frozen['train_sha256']==hashlib.sha256((O/'collection/train.jsonl').read_bytes()).hexdigest()
    assert frozen['readouts_sha256']==hashlib.sha256((e/'readouts.npz').read_bytes()).hexdigest()
    for split,rs in [('train',train),('test',test)]:
        xs=np.load(e/f'{split}-features.npz');history=np.array([[vector(row,t) for t in range(8)] for row in rs]).reshape(len(rs),-1)
        np.testing.assert_allclose(xs['history'],history,rtol=1e-7,atol=1e-7);np.testing.assert_array_equal(xs['labels'],labels(rs))
        response=raw_features(rs)['ledger'];np.testing.assert_allclose(xs['response'],response,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(xs['response_history'],np.concatenate([response,history],1),rtol=1e-7,atol=1e-7)
        fs=np.load(e/f'{split}-forecast.npz')
        actual=np.array([float(vector(row,t)[0]) for row in rs for t in range(1,8)])
        current=np.array([float(vector(row,t)[0]) for row in rs for t in range(7)])
        switches=np.array([frames(row)[t]['target']!=frames(row)[t+1]['target'] for row in rs for t in range(7)])
        np.testing.assert_array_equal(fs['actual'],actual);np.testing.assert_array_equal(fs['persistence'],current);np.testing.assert_array_equal(fs['switched'],switches)
    tx=np.load(e/'train-features.npz');vx=np.load(e/'test-features.npz');arrays=np.load(e/'readouts.npz');y=labels(train)
    groups=np.array([r['repeat'] for r in train]);saved=json.loads((e/'predictions.json').read_text());target=diagnose(labels(test));np.testing.assert_array_equal(saved['target'],target)
    for name,selection in frozen['selection'].items():
        config,counts=choose(tx[name],y,groups);assert config==selection['config'] and counts==selection['cv_exact_correct']
    for name,config in frozen['configs'].items():
        xname='latent' if name=='shuffled_latent' else name
        truth=y[np.random.default_rng(9204).permutation(len(y))] if name=='shuffled_latent' else y
        m=fit(tx[xname],truth,config)
        for k,v in m.items():
            if k!='config':np.testing.assert_allclose(v,arrays[f'{name}_{k}'],rtol=1e-8,atol=1e-9)
        pred=diagnose(score(m,vx[xname]));np.testing.assert_array_equal(pred,saved['methods'][name])
    for name,pred in saved['methods'].items():
        p=np.array(pred);m=summary['methods'][name];assert m['correct']==int(sum(p==target)) and m['n']==32
        for f in range(8):
            mask=np.array([r['fault']==f for r in test]);assert m['families'][str(f)]==dict(n=4,correct=int(sum((p==target)&mask)))
    fs=np.load(e/'test-forecast.npz')
    for name,mask in [('all',np.ones(len(fs['actual']),bool)),('same_target',~fs['switched']),('switched_target',fs['switched'])]:
        for kind in ('predicted','persistence'):assert np.isclose(summary['forecast_mse'][name][kind],np.mean((fs[kind][mask]-fs['actual'][mask])**2))
        assert summary['forecast_mse'][name]['n']==int(sum(mask))
    a=summary['methods']['latent'];reliable=a['correct']>=29 and a['families']['0']['correct']==4 and all(v['correct']>=3 for v in a['families'].values())
    matched=a['correct']>=max(summary['methods'][n]['correct'] for n in ('response','history','response_history'))-1
    fg=summary['forecast_mse']['all']['predicted']<summary['forecast_mse']['all']['persistence']
    assert summary['feasibility_gate']==bool(reliable and matched and fg)
    for name,c in summary['paired'].items():
        a=np.array(saved['methods']['latent'])==target;b=np.array(saved['methods'][name])==target
        wins=int(sum(a&~b));losses=int(sum(b&~a));n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        assert c==dict(latent_only_correct=wins,control_only_correct=losses,p=p)
    meta=json.loads((O/'models/training-91.json').read_text());assert meta['steps']==100000 and meta['seed']==91 and not meta['resumed']
    assert meta['action_contract']=='probe-target-v1' and meta['calibration_sha256']==frozen['train_sha256']
    assert all(json.loads((e/'audit.json').read_text()).values())
    log=(O/'run.log').read_text();assert 'PROBE_PREFLIGHT_PASSED' in log and 'PROBE_SMOKE_PASSED' in log
    assert summary['checkpoint_unchanged'] and summary['readouts_unchanged']
    (O/'verification.json').write_text(json.dumps(dict(source_and_export_hashes=True,training_provenance=True,fresh_balanced_valid_test=True,observation_and_action_alignment=True,readout_selection_and_fits=True,predictions_and_metrics=True,actual_upstream_smoke_and_label_audit=True,latent_and_prior_not_regenerated_locally=True),indent=2)+'\n')
    print('VERIFIED probe-target model evidence',flush=True)
if __name__=='__main__':main()
