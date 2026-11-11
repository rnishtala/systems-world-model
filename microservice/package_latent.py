"""Package only model/config and frozen coefficients; exclude replay and examples."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from microservice.latent_readout import checkpoint_digest,LatentReadout

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=23);a=p.parse_args()
    root=Path(a.models);ev=Path(a.evaluation);out=Path(a.out);seed=a.seed
    fitted=json.loads((ev/'fitted.json').read_text());sha=checkpoint_digest(root/f'final-{seed}')
    if sha!=fitted['checkpoint_sha256']:raise ValueError('Checkpoint no longer matches confirmed readout')
    out.mkdir(parents=True,exist_ok=False);(out/f'train-{seed}').mkdir()
    shutil.copy2(root/f'train-{seed}/config.yaml',out/f'train-{seed}/config.yaml');shutil.copy2(root/f'training-{seed}.json',out/f'training-{seed}.json');shutil.copytree(root/f'final-{seed}',out/f'final-{seed}')
    with np.load(ev/'probes.npz',allow_pickle=False) as data:probe={k:data['latent_'+k] for k in ('mean','scale','weights','bias')}
    np.savez_compressed(out/f'readout-{seed}.npz',**probe)
    meta=dict(seed=seed,checkpoint_sha256=sha,readout_sha256=hashlib.sha256((out/f'readout-{seed}.npz').read_bytes()).hexdigest(),calibration_sha256=fitted['calibration_sha256'],feature_contract='rssm-deter-stoch-mean16-v1',threshold=0.,penalty=1.)
    (out/f'readout-{seed}.json').write_text(json.dumps(meta,indent=2)+'\n');LatentReadout(out,seed)
    inventory=sorted(str(x.relative_to(out)) for x in out.rglob('*') if x.is_file())
    (out.parent/'package.json').write_text(json.dumps(dict(identity=meta,files=inventory),indent=2)+'\n');print('PACKAGED',json.dumps(meta),flush=True)
if __name__=='__main__':main()
