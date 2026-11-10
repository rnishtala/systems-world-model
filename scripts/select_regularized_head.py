"""Reproduce final train-only selection protocol into a fresh output directory."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.regularized_head import select,select_hybrid
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'reports/evidence/latent-probe/train-features.npz'
out=Path(a.out);out.mkdir(parents=True,exist_ok=False);data=np.load(source,allow_pickle=False)
assert len(data['labels'])==140
heads={};selection={}
for name,ranks in [('latent',[8,16,32,64,0]),('raw',[0])]:heads[name],selection[name]=select(data[name],data['labels'],ranks)
np.savez_compressed(out/'heads.npz',**{name+'_'+k:v for name,h in heads.items() for k,v in h.items()})
(out/'selection.json').write_text(json.dumps(selection,indent=2))
h,s=select_hybrid(data['latent'],data['raw'],data['labels'])
np.savez_compressed(out/'hybrid-head.npz',**h);(out/'hybrid-selection.json').write_text(json.dumps(s,indent=2))
(out/'reproduction.json').write_text(json.dumps({'utc':time.time(),'training_features_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Reproduces final amended training-only protocol; no test artifacts read.'},indent=2))
print(json.dumps({**{k:v['chosen'] for k,v in selection.items()},'hybrid':s['chosen']},indent=2))
