"""Run the frozen development experiment and always restore the baseline.
Namespace is fixed. No retry/reuse of failed experiments or alteration of gates.
"""
import hashlib,json,subprocess,time,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/mechanisms'
HOST='rajnishtala@amd-halo';NS='systems-world-model-lab'
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+NS
COLLECT='swm-mechanisms-collect-v1';TRAIN='swm-mechanisms-train-v1';LIVE='swm-mechanisms-live-v1'
def run(command):
    for attempt in range(3):
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,K+' '+command],capture_output=True,text=True,timeout=70)
        if r.returncode==0:return r.stdout
        if r.returncode!=255:break
        time.sleep(2)
    raise RuntimeError(r.stderr[-2500:])
def stage(name,**kw):
    d=dict(stage=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    (OUT/'progress.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def manifest(name,verb='create'):
    p=ROOT/'deploy'/name;doc=json.loads(p.read_text())
    assert all(x['metadata']['namespace']==NS for x in doc.get('items',[doc]))
    subprocess.run(['scp','-o','BatchMode=yes',str(p),HOST+':/tmp/'+name],check=True,timeout=30)
    run(verb+' -f /tmp/'+name)
def wait_job(name,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        job=json.loads(run('get job '+name+' -o json'));s=job.get('status',{})
        if s.get('failed') or s.get('succeeded'):
            (OUT/(name+'.log')).write_text(run('logs job/'+name))
            (OUT/(name+'.json')).write_text(json.dumps({'name':name,'status':s},indent=2))
            if s.get('failed'):raise RuntimeError(name+' failed')
            return
        time.sleep(20)
    raise TimeoutError(name)
def ready():
    deadline=time.monotonic()+480
    while time.monotonic()<deadline:
        ds=json.loads(run('get deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference -o json'))['items']
        if all(d.get('status',{}).get('observedGeneration',0)>=d['metadata']['generation'] and d.get('status',{}).get('availableReplicas',0)==1 and d.get('status',{}).get('updatedReplicas',0)==1 for d in ds):return
        time.sleep(10)
    raise TimeoutError('Deployments not ready')
def export():
    manifest('swm-confirm-reader.json');run('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
    try:
        for remote,local in [('calibration.jsonl','calibration.jsonl'),('live.jsonl','live.jsonl'),('development/training-20.json','training-20.json')]:
            v=run('exec swm-confirm-reader -- cat /results/micro-mechanisms-v1/'+remote)
            for l in v.splitlines() if local.endswith('.jsonl') else [v]:json.loads(l)
            with (OUT/local).open('x') as f:f.write(v)
    finally:run('delete pod swm-confirm-reader --grace-period=1')
try:
    stage('collecting');wait_job(COLLECT,1800)
    stage('training')
    run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=0')
    # Wait for old pods to release quota; deployment scale alone is asynchronous.
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        pods=json.loads(run('get pods -o json'))['items']
        if not any(p.get('status',{}).get('phase') not in ('Succeeded','Failed') for p in pods):break
        time.sleep(5)
    else:raise TimeoutError('Pods did not drain')
    manifest(TRAIN+'.json');wait_job(TRAIN,5400)
    stage('serving_development_model')
    run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1')
    manifest('microservice-agent-mechanisms-v1.json','apply');ready()
    stage('evaluating');manifest(LIVE+'.json');wait_job(LIVE,1800)
    stage('exporting');export()
    env={**os.environ,'PYTHONPATH':str(ROOT)}
    result=subprocess.check_output([sys.executable,str(ROOT/'scripts/mechanisms_summary.py'),str(OUT/'live.jsonl')],text=True,env=env)
    (OUT/'summary.json').write_text(result)
    stage('evaluated',passed=json.loads(result)['passed'])
except Exception as e:
    stage('failed',error=str(e));raise
finally:
    # Terminate only this experiment's active jobs before restoration.
    for name in (COLLECT,TRAIN,LIVE):
        run('delete job '+name+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
    run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1')
    manifest('microservice-agent-confirmation.json','apply');ready()
    (OUT/'restored.json').write_text(json.dumps({'controller':'inspect-first','source':'microservice-source-v10','all_deployments_ready':True},indent=2))
    print('BASELINE_RESTORED',flush=True)
