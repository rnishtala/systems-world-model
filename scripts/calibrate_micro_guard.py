"""Fit frozen empirical residual envelopes from previously completed known cases."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from microservice.guard import residuals
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=Path(a.source);out=Path(a.out)
if out.exists():raise SystemExit('Choose fresh calibration output')
rows=[json.loads(x) for x in source.read_text().splitlines()]
config={'version':'predictive-guard-v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rule':'1.5 * maximum absolute error + 0.05, per seed/true class/sensor','scope':'Earlier confirmation repurposed as calibration for a NEW study; must use fresh validation incidents. No statistical coverage guarantee.','seeds':{}}
for seed in (10,11,12):
 classes={}
 for fault in range(4):
  selected=[r for r in rows if r['method']==seed and r['fault']==fault]
  if len(selected)!=30:raise ValueError('Expected 30 known calibration incidents per class')
  errors=[residuals(r['predictions']) for r in selected]
  classes[str(fault)]={'n':len(selected),'limits':{str(i):1.5*max(e[i] for e in errors)+.05 for i in range(3,10)}}
 config['seeds'][str(seed)]={'classes':classes}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(config,indent=2)+'\n')
print(out)
