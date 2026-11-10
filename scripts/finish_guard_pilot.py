"""Collect the frozen guard pilot and restore the confirmed controller on failure."""
import json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/guard'
HOST='rajnishtala@amd-halo';K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab'
def run(command,check=True):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,K+' '+command],capture_output=True,text=True,timeout=75)
 if check and r.returncode:raise RuntimeError(r.stderr[-2000:])
 return r

def apply(path):
 doc=json.loads(path.read_text());assert all(x['metadata']['namespace']=='systems-world-model-lab' for x in doc.get('items',[doc]))
 subprocess.run(['scp',str(path),HOST+':/tmp/'+path.name],check=True,timeout=30)
 run('apply -f /tmp/'+path.name)

def status(stage,**kw):
 d={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw};(OUT/'progress.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
status('running')
try:
 deadline=time.monotonic()+3600
 while time.monotonic()<deadline:
  job=json.loads(run('get job swm-guard-pilot-v1 -o json').stdout);s=job.get('status',{})
  if s.get('succeeded') or s.get('failed'):break
  time.sleep(20)
 else:raise TimeoutError('Pilot did not finish within one hour')
 (OUT/'job.log').write_text(run('logs job/swm-guard-pilot-v1').stdout)
 status('exporting')
 apply(ROOT/'deploy/swm-confirm-reader.json');run('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
 missing=[]
 for name in ('known','cpu-delay','cpu-pool','pool-delay'):
  r=run(f'exec swm-confirm-reader -- cat /results/micro-guard-v1/{name}.jsonl',check=False)
  if r.returncode:missing.append(name);continue
  target=OUT/(name+'.jsonl')
  if target.exists():raise RuntimeError('Refusing to replace evidence '+str(target))
  for line in r.stdout.splitlines():json.loads(line)
  target.write_text(r.stdout)
 run('delete pod swm-confirm-reader --grace-period=1')
 passed=False
 if not missing and s.get('succeeded'):
  result=subprocess.check_output([sys.executable,str(ROOT/'scripts/guard_summary.py'),str(OUT)],text=True)
  (OUT/'summary.json').write_text(result);passed=json.loads(result)['passed']
 if not passed:
  apply(ROOT/'deploy/microservice-agent-confirmation.json')
  (ROOT/'deploy/microservice-agent.json').write_text((ROOT/'deploy/microservice-agent-confirmation.json').read_text())
 status('complete',passed=passed,missing=missing,deployed_controller='inspect-guarded' if passed else 'inspect-first')
except Exception as e:
 # Preserve the known-working deployment if experiment infrastructure fails.
 try:apply(ROOT/'deploy/microservice-agent-confirmation.json')
 except Exception as restore:status('failed',error=str(e),restore_error=str(restore));raise
 status('failed',error=str(e),deployed_controller='inspect-first');raise
