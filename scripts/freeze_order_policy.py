"""Freeze final readouts/orders on development only before any live confirmation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.order_compare import KEYS,choose_order
from microservice.partial_readout import outer_fit,folds,fit
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1];D=R/'reports/evidence/lab-efficiency-v1/evaluation/development.jsonl';F=R/'reports/evidence/lab-order-v1/evaluation/features.npz';O=R/'reports/evidence/lab-order-confirm-v1/frozen'
def main():
    O.mkdir(parents=True,exist_ok=False);rows=[json.loads(l) for l in D.read_text().splitlines()];assert all(r['seed']==8201 for r in rows)
    families=np.array([r['fault'] for r in rows]);target=np.array([FAULT_MASK[f] for f in families]);y=((target[:,None]>>np.arange(3))&1)*2-1;fs=np.load(F);inner=folds(families,5,8400)
    policies={};arrays={}
    for name in ['latent','raw']:
        runs={key:outer_fit([fs[f'{name}_{key}_{t}'] for t in range(4)],y,families,np.arange(80),np.array([],int),inner) for key in KEYS}
        chosen,threshold,candidates=choose_order(runs,target,families);run=runs[chosen]
        policies[name]=dict(order=chosen,threshold=threshold,readouts=run['selected'],candidates=candidates,selection=run['selection'])
        for t,config in enumerate(run['selected']):
            model=fit(fs[f'{name}_{chosen}_{t}'],y,config)
            for k in ['mean','scale','bias','z','dual']:arrays[f'{name}_{t}_{k}']=model[k]
    np.savez_compressed(O/'readouts.npz',**arrays)
    lp=policies['latent'];selected=next(c for c in lp['candidates'] if c['order']==lp['order'])
    eligible=bool(selected['calibration_passed'] and selected['stopping']['mean_checks']<3)
    identity=json.loads((F.parent/'identity.json').read_text())
    manifest=dict(utc=time.time(),scenario_seed=8401,inner_seed=8400,policies=policies,live_eligible=eligible,checkpoint_sha256=identity['checkpoint_sha256'],development_sha256=hashlib.sha256(D.read_bytes()).hexdigest(),features_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),readouts_sha256=hashlib.sha256((O/'readouts.npz').read_bytes()).hexdigest())
    (O/'policy.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(live_eligible=eligible,policies={name:{k:p[k] for k in ['order','threshold','readouts']} for name,p in policies.items()}),indent=2))
if __name__=='__main__':main()
