"""Reproduce expanded-data readout fits and independent test conclusions."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.probe_range import evaluate,training_cases
from microservice.probe_six import load_readouts
from microservice.probe_target import frames,vector
from microservice.partial_readout import fit
from microservice.latent_probe import labels
from scripts.develop_forecast_readout import choose
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-range-v1'
def main():
    for n,h in json.loads((O/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h,n
    for n,h in json.loads((O/'source-hashes.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    previous,old=load_readouts(O/'frozen')
    for n in ('config.json','readouts.npz'):assert (O/'frozen'/n).read_bytes()==(R/'reports/evidence/probe-six-v1/frozen'/n).read_bytes()
    load=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
    prior=load(O/'collection/train.jsonl');new=load(O/'collection/new-train.jsonl');test=load(O/'evaluation/test.jsonl')
    assert len(prior)==len(new)==len(test)==64
    assert {r['seed'] for r in prior}=={9101,9103} and {r['seed'] for r in new}=={9501} and {r['seed'] for r in test}=={9502}
    for row,c in zip(new,training_cases()):assert (row['fault'],row['repeat'],row['regime'])==(c['fault'],c['repeat'],c['regime'])
    frozen=json.loads((O/'evaluation/frozen.json').read_text())
    for n,h in frozen['train_hashes'].items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h
    assert max(r['utc'] for r in new)<frozen['utc']<min(r['utc'] for r in test)
    assert hashlib.sha256((O/'evaluation/readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256']
    assert frozen['checkpoint_sha256']==previous['checkpoint_sha256']
    for rows in (new,test):
        for row in rows:
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
    for split,rows in [('train',prior+new),('test',test)]:
        for t in (5,7):np.testing.assert_array_equal(arrays[split][f'history_{t}'],[np.concatenate([vector(r,j) for j in range(t+1)]) for r in rows])
    archive=np.load(O/'evaluation/readouts.npz',allow_pickle=False);models={};rows=prior+new;y=labels(rows);groups=np.array([r['repeat']%4 for r in rows])
    for name,selection in frozen['configs'].items():
        x=arrays['train'][f'{name[:-1]}_{int(name[-1])-1}'];config,counts=choose(x,y,groups)
        assert dict(config=config,cv_correct=counts)==selection
        models[name]=dict(config=config,**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith(name+'__')})
        expected=fit(x,y,config)
        for k,v in models[name].items():
            if k!='config':np.testing.assert_allclose(v,expected[k],rtol=1e-8,atol=1e-9)
    result,pred=evaluate(arrays['test'],test,models,old['latent6']);saved=json.loads((O/'evaluation/summary.json').read_text())
    for k,v in result.items():assert saved[k]==v,k
    assert pred==json.loads((O/'evaluation/predictions.json').read_text())
    assert saved['checkpoint_unchanged'] and saved['readouts_unchanged']
    assert json.loads((O/'checkpoint-hashes.json').read_text())==json.loads((R/'reports/evidence/probe-target-model-v1/checkpoint-hashes.json').read_text())
    (O/'verification.json').write_text(json.dumps(dict(source_and_export_hashes_verified=True,training_and_test_seeds_disjoint=True,readouts_frozen_before_test=True,training_fits_reproduced=True,prefix_telemetry_reproduced=True,predictions_and_gates_reproduced=True,world_model_unchanged=True),indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__':main()
