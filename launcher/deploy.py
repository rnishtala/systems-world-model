"""Deploy the isolated lab launcher; preserve ingress rollback state outside Git."""
import subprocess,json,secrets,base64,sys,time,shlex,hashlib
from pathlib import Path
from launcher.manifests import resources
N='systems-world-model-lab'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo']
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+N+' '
def call(args,body=None):
 if '--patch-file=/dev/stdin' in args:
  args=args.replace('--patch-file=/dev/stdin','--patch '+shlex.quote(json.dumps(body)));body=None
 r=subprocess.run(SSH+[K+args],input=json.dumps(body) if body is not None else None,text=True,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return r.stdout
def get(kind,name):return json.loads(call('get '+kind+' '+name+' -o json'))
def apply(obj):print(call('apply -f -',obj).strip(),flush=True)
def secret(name,data,typ='Opaque'):
 try:get('secret',name);return
 except RuntimeError as e:
  if 'NotFound' not in str(e):raise
 apply(dict(apiVersion='v1',kind='Secret',metadata=dict(name=name,namespace=N),type=typ,data={k:base64.b64encode(v).decode() for k,v in data.items()}))
if len(sys.argv)!=3 or sys.argv[1] not in ('backend','ingress','viewer'):
 raise SystemExit('Usage: python -m launcher.deploy backend|ingress|viewer /path/to/images.json')
images=json.loads(Path(sys.argv[2]).read_text())
if sys.argv[1]=='backend':
 secret('swm-experiment-auth',{'key':secrets.token_bytes(32)})
 token=subprocess.check_output(['gh','auth','token'],text=True).strip()
 auth=base64.b64encode(('rnishtala:'+token).encode()).decode()
 secret('swm-experiment-pull',{'.dockerconfigjson':json.dumps({'auths':{'ghcr.io':{'auth':auth}}}).encode()},'kubernetes.io/dockerconfigjson')
 del token,auth
 apply(resources(images))
 print(call('rollout status deployment/swm-experiments --timeout=150s'))
elif sys.argv[1]=='ingress':
 dep=get('deployment','swm-headlamp-public');cm=get('configmap','swm-ngrok-agent')
 backup=Path('/tmp/swm-launcher-rollback.json')
 if not backup.exists():backup.write_text(json.dumps({'deployment':dep,'configmap':cm}))
 spec=dep['spec']['template']['spec']
 metadata=dep['spec']['template'].get('metadata',{})
 plugin=next(i for i in resources(images)['items'] if i['kind']=='ConfigMap')
 metadata.setdefault('annotations',{})['swm-experiments/plugin-sha256']=hashlib.sha256(json.dumps(plugin['data'],sort_keys=True).encode()).hexdigest()
 sc=dict(allowPrivilegeEscalation=False,readOnlyRootFilesystem=True,capabilities=dict(drop=['ALL']))
 proxy=dict(name='experiment-proxy',image=images['control'],command=['python','-m','launcher.proxy'],resources=dict(requests=dict(cpu='5m',memory='32Mi'),limits=dict(cpu='20m',memory='64Mi')),securityContext=sc,volumeMounts=[dict(name='experiment-auth',mountPath='/auth',readOnly=True)],readinessProbe=dict(exec=dict(command=['python','-c','import urllib.request; urllib.request.urlopen("http://127.0.0.1:4467/healthz",timeout=2)']),initialDelaySeconds=3,timeoutSeconds=3))
 spec['containers']=[c for c in spec['containers'] if c['name']!='experiment-proxy']+[proxy]
 head=next(c for c in spec['containers'] if c['name']=='headlamp')
 head['volumeMounts']=[v for v in head['volumeMounts'] if v['name']!='experiment-plugin']+[dict(name='experiment-plugin',mountPath='/headlamp/static-plugins/swm-experiments',readOnly=True)]
 spec['volumes']=[v for v in spec['volumes'] if v['name'] not in ('experiment-auth','experiment-plugin')]+[dict(name='experiment-auth',secret=dict(secretName='swm-experiment-auth')),dict(name='experiment-plugin',configMap=dict(name='swm-headlamp-experiments'))]
 spec['imagePullSecrets']=[dict(name='swm-experiment-pull')]
 config=json.loads(cm['data']['ngrok.yml']);config['endpoints'][0]['upstream']['url']='http://127.0.0.1:4467'
 apply(dict(apiVersion='v1',kind='ConfigMap',metadata=dict(name='swm-ngrok-agent',namespace=N),data={'ngrok.yml':json.dumps(config)}))
 try:
  print(call('patch deployment swm-headlamp-public --type=merge --patch-file=/dev/stdin',{'spec':{'template':{'spec':spec,'metadata':metadata}}}))
  print(call('rollout status deployment/swm-headlamp-public --timeout=150s'))
 except Exception:
  old=json.loads(backup.read_text());apply(dict(apiVersion='v1',kind='ConfigMap',metadata=dict(name='swm-ngrok-agent',namespace=N),data=old['configmap']['data']))
  call('patch deployment swm-headlamp-public --type=merge --patch-file=/dev/stdin',{'spec':old['deployment']['spec']})
  raise
elif sys.argv[1]=='viewer':
 binding=get('rolebinding','swm-headlamp-public')
 if binding['roleRef']['name']!='swm-ui-viewer':
  call('delete rolebinding swm-headlamp-public')
  binding['roleRef']['name']='swm-ui-viewer'
  apply({k:binding[k] for k in ['apiVersion','kind','roleRef','subjects']}|{'metadata':dict(name='swm-headlamp-public',namespace=N)})
 result=subprocess.run(SSH+[K+'auth can-i create jobs --as=system:serviceaccount:'+N+':swm-headlamp-public'],capture_output=True,text=True,timeout=30)
 if result.stdout.strip()!='no':raise RuntimeError('Public Headlamp write restriction not verified')
 print('Public Headlamp Job creation denied (expected)')
