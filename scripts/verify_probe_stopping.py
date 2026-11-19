"""Check exported integrity, prefix-only telemetry, and saved stopping outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.evaluate_probe_stopping import O,R,decide,metrics,NAMES
from microservice.probe_target import vector,frames
from microservice.latent_probe import labels

def main():
    for file,sha in json.loads((O/'frozen-hashes.json').read_text()).items():
        assert hashlib.sha256((O/file).read_bytes()).hexdigest()==sha,file
    for file,sha in json.loads((O/'analysis-input-hashes.json').read_text()).items():
        assert hashlib.sha256((R/file).read_bytes()).hexdigest()==sha,file
    rows=[json.loads(l) for l in (O/'collection/train.jsonl').read_text().splitlines()]
    fs=np.load(O/'evaluation/prefix-features.npz',allow_pickle=False)
    for t in range(8):
        expected=[]
        for row in rows:
            seq=np.array([vector(row,j) for j in range(t+1)]);summary=[]
            for target in (0,1):
                for load in (.25,.75):
                    a=seq[(seq[:,4]==target)&np.isclose(seq[:,5],load)]
                    summary.extend(a[:,:4].mean(0) if len(a) else np.zeros(4));summary.extend([len(a),bool(len(a))])
            expected.append(np.r_[summary,seq.reshape(-1)])
        np.testing.assert_array_equal(fs[f'summary_history_{t}'],expected)
    y=labels(rows);families=np.array([r['fault'] for r in rows]);cost=np.array([np.cumsum([f['requests'] for f in frames(r)]) for r in rows])
    saved=json.loads((O/'predictions.json').read_text());s=json.loads((O/'summary.json').read_text());folds=json.loads((O/'splits.json').read_text())
    for name in NAMES:
        p=saved['methods'][name];values=np.array(p['scores']);pred=np.empty(64,int);stop=np.empty(64,int)
        for fold,detail in zip(folds,p['folds']):
            held=np.array(fold['held']);assert not set(held)&set(fold['train'])
            pred[held],stop[held]=decide(values[held],detail['policy']['threshold'])
        np.testing.assert_array_equal(pred,p['predictions']);np.testing.assert_array_equal(stop,p['stops'])
        assert metrics(pred,stop,y,families,cost)==s['methods'][name]['stopping']
    extraction=json.loads((O/'evaluation/extraction.json').read_text());assert extraction['checkpoint_unchanged']
    previous=json.loads((R/'reports/evidence/probe-target-model-v1/evaluation/frozen.json').read_text())
    assert extraction['checkpoint_sha256']==previous['checkpoint_sha256']
    assert json.loads((O/'checkpoint-hashes.json').read_text())==json.loads((R/'reports/evidence/probe-target-model-v1/checkpoint-hashes.json').read_text())
    audit=json.loads((O/'evaluation/audit.json').read_text())
    (O/'verification.json').write_text(json.dumps(dict(export_hashes_verified=True,analysis_inputs_verified=True,prefix_telemetry_reproduced=True,stopping_decisions_reproduced=True,checkpoint_matches_previous=True,audit=audit),indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__':main()
