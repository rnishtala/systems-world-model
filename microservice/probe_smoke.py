import json
from pathlib import Path
from microservice.agent import Model
from microservice.auxiliary_audit import audit
from microservice.probe_target import extract
model=Model(str(Path('smoke').resolve()),91,'native');assert all(audit(model).values())
rows=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()[:2]]
x,aux,f=extract(model,rows)
assert x['latent'].shape==(2,640) and x['history'].shape==(2,112) and f['actual'].shape==(14,)
try:model.act([0]*14,True,0,'unused')
except ValueError as e:assert 'Offline' in str(e)
else:raise AssertionError('Offline model accepted serving API call')
print('PROBE_SMOKE_PASSED checkpoint, posterior, prior, label audit and offline-only guard',flush=True)
