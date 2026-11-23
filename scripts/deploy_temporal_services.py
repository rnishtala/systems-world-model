"""Roll out the opt-in temporal instrumentation to idle lab services only."""
import json,subprocess,hashlib,shlex
from pathlib import Path
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo']
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab '
def call(cmd,body=None):
 if '--patch-file=/dev/stdin' in cmd:
  cmd=cmd.replace('--patch-file=/dev/stdin','--patch '+shlex.quote(json.dumps(body)));body=None
 r=subprocess.run(SSH+[K+cmd],input=json.dumps(body) if body is not None else None,text=True,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr[-1500:])
 return r.stdout
check="exec deployment/swm-experiments -- python -c 'from launcher.store import Store; from launcher.common import TERMINAL; assert all(r[\"status\"] in TERMINAL for r in Store(\"/results/experiments/queue.sqlite\").all()); print(\"Lab idle\")'"
print(call(check),flush=True)
source=(Path(__file__).resolve().parents[1]/'microservice/service.py').read_text();sha=hashlib.sha256(source.encode()).hexdigest();name='swm-temporal-service-'+sha[:12]
print(call('apply -f -',dict(apiVersion='v1',kind='ConfigMap',metadata=dict(name=name,namespace='systems-world-model-lab'),immutable=True,data={'service.py':source})),flush=True)
for role in ('checkout','payment','ledger'):
 name_dep='swm-'+role;dep=json.loads(call('get deployment '+name_dep+' -o json'));spec=dep['spec']['template']['spec']
 init=spec['initContainers'][0]
 if init['command'][:2]!=['sh','-ec'] or not init['command'][2].startswith('tar xzf /source/app.tar.gz -C /app'):raise ValueError('Unexpected service initialization')
 init['command']=['sh','-ec','tar xzf /source/app.tar.gz -C /app && cp /temporal/service.py /app/microservice/service.py']
 init['volumeMounts']=[v for v in init['volumeMounts'] if v['name']!='temporal-service']+[dict(name='temporal-service',mountPath='/temporal',readOnly=True)]
 spec['volumes']=[v for v in spec['volumes'] if v['name']!='temporal-service']+[dict(name='temporal-service',configMap=dict(name=name))]
 patch={'spec':{'template':{'spec':{'initContainers':spec['initContainers'],'volumes':spec['volumes']},'metadata':{'annotations':{'swm/temporal-service-sha256':sha}}}}}
 print(call('patch deployment '+name_dep+' --type=merge --patch-file=/dev/stdin',patch),flush=True)
 print(call('rollout status deployment/'+name_dep+' --timeout=150s'),flush=True)
