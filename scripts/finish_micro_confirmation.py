"""Finish the frozen experiment; all cluster operations target one fixed namespace.
Run after the three swm-confirm-*-v1 training Jobs have been created.
No retries of failed experiments, configuration tuning, or Git mutations.
"""
import json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NS='systems-world-model-lab';HOST='rajnishtala@amd-halo'
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml'
EVIDENCE=ROOT/'reports/evidence/microservice'
STATE=EVIDENCE/'confirmation-progress.json'
def stage(name,**kw):
 data=dict(stage=name,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 STATE.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)
def remote(command):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command],text=True,capture_output=True,timeout=75)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def kube(args):return remote(f'{K} -n {NS} {args}')
def upload(path):
 data=json.loads(path.read_text());items=data.get('items',[data])
 assert all(x['metadata']['namespace']==NS for x in items)
 subprocess.run(['scp',str(path),f'{HOST}:/tmp/{path.name}'],check=True,timeout=30)
 return '/tmp/'+path.name
def wait_jobs(names,timeout):
 deadline=time.monotonic()+timeout;last=None
 while time.monotonic()<deadline:
  response=json.loads(kube('get jobs '+' '.join(names)+' -o json'));jobs=response.get('items',[response])
  statuses={j['metadata']['name']:j.get('status',{}) for j in jobs}
  if any(s.get('failed',0)>0 for s in statuses.values()):raise RuntimeError('Experiment failed: '+json.dumps(statuses))
  if all(s.get('succeeded',0)==1 for s in statuses.values()):return
  update={n:('complete' if s.get('succeeded') else 'running') for n,s in statuses.items()}
  if update!=last:stage('waiting',jobs=update);last=update
  time.sleep(20)
 raise TimeoutError('Experiment deadline exceeded')
def export(remote_path,local_name):
 target=EVIDENCE/local_name
 if target.exists():raise RuntimeError('Refusing to overwrite '+str(target))
 value=kube(f'exec swm-confirm-reader -- cat {remote_path}')
 # Reject incomplete or invalid JSON before preserving it.
 if local_name.endswith('.jsonl'):
  for line in value.splitlines():json.loads(line)
 else:json.loads(value)
 target.write_text(value)
try:
 stage('waiting_for_training')
 wait_jobs([f'swm-confirm-{s}-v1' for s in (10,11,12)],5400)
 stage('restoring_lab')
 kube('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1')
 manifest=upload(ROOT/'deploy/microservice-agent-confirmation.json');kube('apply -f '+manifest)
 deadline=time.monotonic()+360
 while time.monotonic()<deadline:
  d=json.loads(kube('get deployment swm-inference -o json'))
  if d.get('status',{}).get('availableReplicas',0)==1:break
  time.sleep(10)
 else:raise TimeoutError('Inference did not become ready')
 stage('live_confirmation')
 manifest=upload(ROOT/'deploy/swm-live-confirm-v1.json');kube('apply -f '+manifest)
 wait_jobs(['swm-live-confirm-v1'],3600)
 stage('exporting_evidence')
 manifest=upload(ROOT/'deploy/swm-confirm-reader.json');kube('apply -f '+manifest)
 kube('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
 export('/results/micro-v2/confirmation-live.jsonl','confirmation-live.jsonl')
 for seed in (10,11,12):export(f'/results/micro-v2/confirmation/training-{seed}.json',f'confirmation-training-{seed}.json')
 kube('delete pod swm-confirm-reader --grace-period=1')
 summary=subprocess.check_output([sys.executable,str(ROOT/'scripts/micro_summary.py'),'--calibration',str(EVIDENCE/'calibration.jsonl'),'--results',str(EVIDENCE/'confirmation-live.jsonl'),'--confirmation'],text=True)
 (EVIDENCE/'confirmation-summary.json').write_text(summary)
 result=json.loads(summary)
 stage('complete',gate_passed=result['frozen_confirmation_passed'],physical_incidents=result['physical_incidents'])
except Exception as e:
 stage('failed',error=str(e));raise
