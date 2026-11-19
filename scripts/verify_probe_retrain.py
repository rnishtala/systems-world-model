"""Reproduce expanded-data readout fits and independent test conclusions."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.probe_retrain import evaluate
from microservice.probe_six import load_readouts
from microservice.probe_target import frames,vector
from microservice.partial_readout import fit
from microservice.latent_probe import labels
from scripts.develop_forecast_readout import choose
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-retrain-v1'
def main():
    for n,h in json.loads((O/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h,n
    for n,h in json.loads((O/'source-hashes.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    previous,old=load_readouts(O/'frozen')
    for n in ('config.json','readouts.npz'):assert (O/'frozen'/n).read_bytes()==(R/'reports/evidence/probe-six-v1/frozen'/n).read_bytes()
    load=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
    rows=load(O/'collection/train.jsonl');test=load(O/'evaluation/test.jsonl')
    previous_root=R/'reports/evidence/probe-range-v1'
    expected=(previous_root/'collection/train.jsonl').read_bytes()+(previous_root/'collection/new-train.jsonl').read_bytes()
    assert (O/'collection/train.jsonl').read_bytes()==expected
    assert len(rows)==128 and len(test)==64
    assert {r['seed'] for r in rows}=={9101,9103,9501} and {r['seed'] for r in test}=={9503}
    for src,dst in [('frozen.json','refit-config.json'),('readouts.npz','refit-readouts.npz')]:assert (previous_root/'evaluation'/src).read_bytes()==(O/'frozen'/dst).read_bytes()
    frozen=json.loads((O/'evaluation/frozen.json').read_text())
    assert hashlib.sha256(expected).hexdigest()==frozen['train_sha256']
    assert max(r['utc'] for r in rows)<frozen['utc']<min(r['utc'] for r in test)
    assert hashlib.sha256((O/'evaluation/readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256']
    assert frozen['checkpoint_sha256']!=previous['checkpoint_sha256']
    for measured in (test,):
        for row in measured:
            assert row['seconds']<=35
            assert row['order']==(['checkout','ledger'] if row['repeat']%4<2 else ['ledger','checkout'])
            expected=[3,9] if row['regime']==0 else [4,10]
            if row['repeat']%2:expected.reverse()
            assert [f['requests'] for f in frames(row)]==expected*4
            for f in frames(row):
                for service,m in f['metrics'].items():
                    count=f['requests'] if f['target']=='checkout' or service=='ledger' else 0
                    assert m['count']==count and m['error_rate']==0
    for g in (0,1):
        for fault in range(8):assert sum(r['regime']==g and r['fault']==fault for r in test)==4
    arrays={split:np.load(O/f'evaluation/{split}-features.npz',allow_pickle=False) for split in ('train','test')}
    for split,subset in [('train',rows),('test',test)]:
        for t in (5,7):np.testing.assert_array_equal(arrays[split][f'history_{t}'],[np.concatenate([vector(r,j) for j in range(t+1)]) for r in subset])
    archive=np.load(O/'evaluation/readouts.npz',allow_pickle=False);models={};y=labels(rows);groups=np.array([r['repeat']%4 for r in rows])
    for name,selection in frozen['configs'].items():
        x=arrays['train'][f'{name[:-1]}_{int(name[-1])-1}'];config,counts=choose(x,y,groups)
        assert dict(config=config,cv_correct=counts)==selection
        models[name]=dict(config=config,**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith(name+'__')})
        expected=fit(x,y,config)
        for k,v in models[name].items():
            if k!='config':np.testing.assert_allclose(v,expected[k],rtol=1e-8,atol=1e-9)
    refit_config=json.loads((O/'frozen/refit-config.json').read_text())
    archive=np.load(O/'frozen/refit-readouts.npz',allow_pickle=False)
    refit=dict(config=refit_config['configs']['latent6']['config'],**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith('latent6__')})
    old_x=np.load(O/'evaluation/baseline-features.npz',allow_pickle=False)['latent_5']
    result,pred=evaluate(arrays['test'],test,models,old['latent6'],refit,old_x);saved=json.loads((O/'evaluation/summary.json').read_text())
    for k,v in result.items():assert saved[k]==v,k
    assert pred==json.loads((O/'evaluation/predictions.json').read_text())
    assert saved['checkpoints_unchanged_after_freeze'] and saved['world_model_retrained']
    training=json.loads((O/'models/training-101.json').read_text())
    assert training['steps']==100000 and training['seed']==101 and not training['resumed']
    assert training['calibration_sha256']==frozen['train_sha256']
    assert all(json.loads((O/'evaluation/audit.json').read_text()).values())
    assert json.loads((O/'restored.json').read_text())['template_unchanged']
    (O/'verification.json').write_text(json.dumps(dict(source_and_export_hashes_verified=True,designated_training_bytes_reused=True,training_and_test_seeds_disjoint=True,readouts_frozen_before_test=True,training_fits_reproduced=True,prefix_telemetry_reproduced=True,predictions_and_gates_reproduced=True,world_model_retrained=True),indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__':main()
