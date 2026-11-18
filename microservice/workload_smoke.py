"""Load the short upstream checkpoint and exercise posterior and prior forecasts."""
import json
from pathlib import Path
import numpy as np
from microservice.agent import Model
from microservice.auxiliary_audit import audit
from microservice.workload import extract
model=Model(str(Path('smoke').resolve()),81,'native');assert all(audit(model).values())
rows=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()[:2]]
f,a,p=extract(model,rows)
assert f['latent'].shape==(2,640) and p['actual'].shape==(6,4)
assert all(np.isfinite(v).all() for v in list(f.values())+list(p.values()))
print('WORKLOAD_SMOKE_PASSED checkpoint, label isolation, continuous action posterior and prior',flush=True)
