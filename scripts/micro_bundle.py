import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--deps',default='/tmp/swm-micro-pg-deps');a=p.parse_args()
with tarfile.open(a.output,'w:gz') as t:
 def clean(i):return None if '__pycache__' in i.name or '.pytest_cache' in i.name else i
 for name in ('microservice','experiment','tests','deploy/microservice-lab.json'):t.add(name,arcname=name,filter=clean)
 t.add(a.deps,arcname='deps',filter=clean)
print(json.dumps({'file':a.output,'sha256':hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),'bytes':Path(a.output).stat().st_size}))
