"""Bounded standalone probe workflow with baseline restoration."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/latent-probe';O.mkdir(parents=True,exist_ok=True)
HOST='rajnishtala@amd-halo';K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab';JOB='swm-latent-probe-v1'
def remote(cmd):
 for i in range(3):
  p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,K+' '+cmd],capture_output=True,timeout=70)
  if p.returncode==0:return p.stdout
  if p.returncode!=255:break
  time.sleep(2)
 raise RuntimeError(p.stderr.decode()[-2500:])
def stage(s):
 d={'stage':s,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(O/'progress.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
try:
 stage('starting')
 remote('create configmap microservice-source-v14 --from-file=app.tar.gz=/tmp/swm-micro-v14.tar.gz')
 remote('scale deployment swm-inference --replicas=0')
 remote('wait --for=delete pod -l swm-component=inference --timeout=55s')
 remote('create -f /tmp/swm-latent-probe-v1.json')
 stage('running');deadline=time.monotonic()+2400
 while time.monotonic()<deadline:
  j=json.loads(remote('get job '+JOB+' -o json'));s=j.get('status',{})
  if s.get('succeeded') or s.get('failed'):break
  time.sleep(20)
 else:raise TimeoutError('Probe job deadline')
 (O/'job.log').write_bytes(remote('logs job/'+JOB));(O/'job-status.json').write_text(json.dumps(s,indent=2))
 if s.get('failed'):raise RuntimeError('Probe job failed; see job.log')
 stage('exporting')
 p=R/'deploy/swm-confirm-reader.json';subprocess.run(['scp',str(p),HOST+':/tmp/'+p.name],check=True,timeout=30)
 remote('create -f /tmp/'+p.name);remote('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
 try:
  for name in ('fitted.json','summary.json','predictions.json','test.jsonl','train-features.npz','test-features.npz','probes.npz'):
   value=remote('exec swm-confirm-reader -- cat /results/micro-latent-v1/'+name)
   with (O/name).open('xb') as f:f.write(value)
 finally:remote('delete pod swm-confirm-reader --grace-period=1')
 stage('evaluated')
finally:
 remote('delete job '+JOB+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
 remote('scale deployment swm-inference --replicas=1')
 deadline=time.monotonic()+480
 while time.monotonic()<deadline:
  d=json.loads(remote('get deployment swm-inference -o json'))
  if d.get('status',{}).get('availableReplicas')==1:break
  time.sleep(10)
 else:raise TimeoutError('Restoration failed')
 (O/'restored.json').write_text(json.dumps({'inference_ready':True,'controller':'inspect-first','source':'microservice-source-v10'},indent=2))
 print('BASELINE_RESTORED',flush=True)
