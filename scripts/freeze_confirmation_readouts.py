"""Freeze readouts using training-only workload-group selection."""
import hashlib,json,time
import numpy as np
from scripts.develop_forecast_readout import R,SOURCE,AUDIT,choose
from microservice.forecast_readout import features
from microservice.latent_probe import labels
from microservice.partial_readout import fit

OUT=R/'reports/evidence/confirmation-readouts-v1'
def main():
    OUT.mkdir(exist_ok=False)
    paths=[SOURCE/'collection/train.jsonl',SOURCE/'evaluation/train-features.npz',AUDIT/'train.npz']
    rows=[json.loads(l) for l in paths[0].read_text().splitlines()]
    assert len(rows)==128 and all(r['seed']==8701 for r in rows)
    xs=features(np.load(paths[1]),np.load(paths[2]));y=labels(rows)
    groups=np.array([r['regime'] for r in rows]);models={};selection={}
    for name,x in xs.items():
        config,counts=choose(x,y,groups)
        models[name]=fit(x,y,config);selection[name]=dict(config=config,cv_correct=counts)
    np.savez_compressed(OUT/'readouts.npz',**{f'{n}_{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
    frozen=dict(utc=time.time(),selection=selection,
        checkpoint_sha256=json.loads((AUDIT/'summary.json').read_text())['checkpoint_before'],
        readouts_sha256=hashlib.sha256((OUT/'readouts.npz').read_bytes()).hexdigest(),
        input_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        primary='latent',confirmation_seed=9002,refit_after_collection=False)
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('READOUTS_FROZEN',frozen['readouts_sha256'])
if __name__=='__main__':main()
