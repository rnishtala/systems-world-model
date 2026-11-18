import json,subprocess,time
from decimal import Decimal
from pathlib import Path
R=Path("/Users/rnishtala/src/systems-world-model");O=R/"reports/evidence/rcaeval-alignment-v1";O.mkdir(parents=True,exist_ok=False)
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo','rajnishtala@amd-halo'];K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab '
def run(cmd,timeout=65):
 attempts=5 if cmd.startswith(('get ','logs ','rollout ','scale ','delete ')) else 1
 error=None
 for attempt in range(attempts):
  try:
   p=subprocess.run(SSH+[K+cmd],capture_output=True,text=True,timeout=timeout)
   if p.returncode:raise RuntimeError(p.stderr[-1500:])
   return p.stdout
  except (RuntimeError,subprocess.TimeoutExpired) as exc:
   error=exc
   if attempt+1<attempts:time.sleep(2)
 raise error
subprocess.run(['scp',*SSH[1:-1],str(R/'deploy/rcaeval-alignment-v1.json'),SSH[-1]+':/tmp/rcaeval-alignment-v1.json'],check=True,timeout=30)
subprocess.run(['scp',*SSH[1:-1],str(R/'deploy/rcaeval-stage-v1.json'),SSH[-1]+':/tmp/rcaeval-stage-v1.json'],check=True,timeout=30)
active=[j['metadata']['name'] for j in json.loads(run('get jobs -o json'))['items'] if j.get('status',{}).get('active',0)];assert not active,active
print(run('create -f /tmp/rcaeval-stage-v1.json'),flush=True);print(run('wait pod/swm-rcaeval-stage-v1 --for=condition=Ready --timeout=55s'),flush=True)
try:
 print(run('exec swm-rcaeval-stage-v1 -- mkdir /results/rcaeval-alignment-v1'))
 with open('/tmp/rcaeval-alignment-v1.tar.gz','rb') as f:subprocess.run(SSH+[K+'exec -i swm-rcaeval-stage-v1 -- tar xzf - -C /results/rcaeval-alignment-v1'],stdin=f,check=True,timeout=90)
finally:print(run('delete pod swm-rcaeval-stage-v1 --grace-period=1 --wait=true --timeout=55s --ignore-not-found=true'),flush=True)
before=json.loads(run('get deployment swm-inference -o json'));(O/'before.json').write_text(json.dumps(before['spec'],indent=2));paused=False;created=False
try:
 print(run('scale deployment swm-inference --replicas=0'),flush=True);paused=True
 for _ in range(30):
  q=json.loads(run('get resourcequota lab-budget -o json'))
  # Leave 2 CPU for training; account for the persistent lab UI.
  cpu=q['status']['used']['limits.cpu']
  used=Decimal(cpu[:-1])/1000 if cpu.endswith('m') else Decimal(cpu)
  if used <= 2:break
  time.sleep(2)
 else:raise RuntimeError('Quota not released')
 print(run('create -f /tmp/rcaeval-alignment-v1.json'),flush=True);created=True
 for _ in range(185):
  status=json.loads(run('get job swm-rcaeval-alignment-v1 -o json'))['status']
  if status.get('failed') or status.get('succeeded'):
   (O/'job.json').write_text(json.dumps(status,indent=2));(O/'run.log').write_text(run('logs job/swm-rcaeval-alignment-v1'))
   if status.get('failed'):raise RuntimeError('Readout failed')
   break
  print('READOUT_ACTIVE',flush=True);time.sleep(20)
 else:raise TimeoutError('Readout deadline')
finally:
 try:
  if created:print(run('delete job swm-rcaeval-alignment-v1 --wait=true --timeout=55s --ignore-not-found=true'),flush=True)
 finally:
  if paused:
   print(run('scale deployment swm-inference --replicas='+str(before['spec']['replicas'])),flush=True);print(run('rollout status deployment/swm-inference --timeout=55s'),flush=True)
   after=json.loads(run('get deployment swm-inference -o json'));assert after['spec']['template']==before['spec']['template'];(O/'restored.json').write_text(json.dumps({'template_unchanged':True,'replicas':after['spec']['replicas'],'ready':True}))
print(run('create -f /tmp/rcaeval-stage-v1.json'),flush=True);print(run('wait pod/swm-rcaeval-stage-v1 --for=condition=Ready --timeout=55s'),flush=True)
try:

 code="from pathlib import Path\nimport json,hashlib\nr=Path('/results/rcaeval-alignment-v1')\npaths=list(r.glob('audit-*/*.json'))\nfiles={str(p.relative_to(r)):p.read_text() for p in paths}\nhashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}\nmodel_hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('model-*/checkpoint/**/*') if p.is_file()}\nfiles['checkpoint-hashes.json']=json.dumps(model_hashes,indent=2)\nprint(json.dumps(dict(files=files,hashes=hashes)))"
 data=json.loads(subprocess.run(SSH+[K+'exec -i swm-rcaeval-stage-v1 -- /results/venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=65).stdout)
 for name,content in data['files'].items():
  path=O/name;path.parent.mkdir(exist_ok=True);path.write_text(content)
 (O/'frozen-hashes.json').write_text(json.dumps(data['hashes'],indent=2))
finally:print(run('delete pod swm-rcaeval-stage-v1 --grace-period=1 --wait=true --timeout=55s --ignore-not-found=true'),flush=True)
print('READOUT_EXPORTED',flush=True)
