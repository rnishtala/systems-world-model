"""Reproduce frozen fits, prefix telemetry, and fresh confirmation scoring."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.probe_six import load_readouts,score_rows
from microservice.probe_target import vector
from microservice.latent_probe import labels
from microservice.partial_readout import fit
from scripts.develop_forecast_readout import choose
from scripts.evaluate_ledger_probe import features
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-six-v1'

def main():
    for name,sha in json.loads((O/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==sha,name
    for name,sha in json.loads((O/'source-hashes.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
    cfg,models=load_readouts(O/'frozen');train=[json.loads(l) for l in (O/'collection/train.jsonl').read_text().splitlines()]
    assert hashlib.sha256((O/'collection/train.jsonl').read_bytes()).hexdigest()==cfg['train_sha256']
    assert hashlib.sha256((R/'reports/PROBE_SIX_PROTOCOL.md').read_bytes()).hexdigest()==cfg['protocol_sha256']
    src=R/'reports/evidence/probe-stopping-v1/evaluation/prefix-features.npz'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==cfg['training_features_sha256']
    fs=np.load(src,allow_pickle=False);xs={f'{n}{t+1}':fs[f'{n}_{t}'] for n in ('latent','history','summary_history') for t in (5,7)};xs['response8']=features(train)['ledger']
    y=labels(train);groups=np.array([r['repeat'] for r in train])
    for n,x in xs.items():
        c,counts=choose(x,y,groups);assert dict(config=c,cv_correct=counts)==cfg['configs'][n]
        expected=fit(x,y,c)
        for k,v in models[n].items():
            if k!='config':np.testing.assert_allclose(v,expected[k],rtol=1e-8,atol=1e-9)
    rows=[json.loads(l) for l in (O/'evaluation/test.jsonl').read_text().splitlines()]
    assert len(rows)==32 and {r['seed'] for r in rows}=={9302}
    assert all(sum(r['fault']==f for r in rows)==4 for f in range(8))
    assert all(r['seconds']<=35 and r['utc']>cfg['utc'] for r in rows)
    from microservice.probe_target import frames
    for row in rows:
        assert row['order']==(['checkout','ledger'] if row['repeat']<2 else ['ledger','checkout'])
        assert sum(f['requests'] for f in frames(row)[:6])==36
        assert sum(f['requests'] for f in frames(row))==48
        for frame in frames(row):
            for service,m in frame['metrics'].items():
                expected=frame['requests'] if frame['target']=='checkout' or service=='ledger' else 0
                assert m['count']==expected and m['error_rate']==0
    xs=np.load(O/'evaluation/test-features.npz',allow_pickle=False)
    for t in range(8):
        expected=[]
        for row in rows:
            seq=np.array([vector(row,j) for j in range(t+1)]);summary=[]
            for target in (0,1):
                for load in (.25,.75):
                    seen=seq[(seq[:,4]==target)&np.isclose(seq[:,5],load)]
                    summary.extend(seen[:,:4].mean(0) if len(seen) else np.zeros(4));summary.extend([len(seen),bool(len(seen))])
            expected.append(np.r_[summary,seq.reshape(-1)])
        np.testing.assert_array_equal(xs[f'summary_history_{t}'],expected)
        np.testing.assert_array_equal(xs[f'history_{t}'],[np.concatenate([vector(row,j) for j in range(t+1)]) for row in rows])
    result,pred=score_rows(xs,rows,models);saved=json.loads((O/'evaluation/summary.json').read_text())
    assert pred==json.loads((O/'evaluation/predictions.json').read_text())
    for k,v in result.items():assert saved[k]==v,k
    assert saved['collection_started_utc']>saved['frozen_utc']==cfg['utc']
    assert saved['checkpoint_unchanged'] and saved['readouts_unchanged']
    assert json.loads((O/'checkpoint-hashes.json').read_text())==json.loads((R/'reports/evidence/probe-target-model-v1/checkpoint-hashes.json').read_text())
    (O/'verification.json').write_text(json.dumps(dict(export_and_source_hashes_verified=True,training_only_fits_reproduced=True,prefix_telemetry_reproduced=True,predictions_and_gate_reproduced=True,frozen_before_collection=True,checkpoint_matches_previous=True),indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__':main()
