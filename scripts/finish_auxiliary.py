"""Run the frozen development experiment and always restore the baseline.
Namespace is fixed. No retry/reuse of failed experiments or alteration of gates.
"""
import hashlib,json,subprocess,time,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/auxiliary';OUT.mkdir(parents=True,exist_ok=True)
HOST='rajnishtala@amd-halo';NS='systems-world-model-lab'
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+NS
TRAIN='swm-auxiliary-train-v2';LIVE='swm-auxiliary-eval-v1'
def run(command):
    for attempt in range(30):
        try:r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo',HOST,K+' '+command],capture_output=True,text=True,timeout=70)
        except subprocess.TimeoutExpired:
            time.sleep(2);continue
        if r.returncode==0:return r.stdout
        if r.returncode!=255:raise RuntimeError(r.stderr[-2500:])
        time.sleep(2)
    raise ConnectionError('SSH retries exhausted; remote state must be checked before cleanup')
def stage(name,**kw):
    d=dict(stage=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    (OUT/'progress.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def manifest(name,verb='create'):
    p=ROOT/'deploy'/name;doc=json.loads(p.read_text())
    assert all(x['metadata']['namespace']==NS for x in doc.get('items',[doc]))
    subprocess.run(['scp','-o','BatchMode=yes','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo',str(p),HOST+':/tmp/'+name],check=True,timeout=30)
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
if __name__ == '__main__':
    connectivity_lost=False
    try:
        stage('starting')
        subprocess.run(['scp','-o','BatchMode=yes','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo','/tmp/swm-micro-v16.tar.gz',HOST+':/tmp/swm-micro-v16.tar.gz'],check=True,timeout=30)
        run('create configmap microservice-source-v16 --from-file=app.tar.gz=/tmp/swm-micro-v16.tar.gz')
        run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=0')
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            pods=json.loads(run('get pods -o json'))['items']
            if not any(p.get('status',{}).get('phase') not in ('Succeeded','Failed') for p in pods):break
            time.sleep(5)
        else:raise TimeoutError('Pods did not drain')
        manifest(TRAIN+'.json');stage('training');wait_job(TRAIN,2400)
        run('delete job '+TRAIN+' --cascade=foreground --wait=true --timeout=55s')
        run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1')
        manifest(LIVE+'.json');stage('evaluating');wait_job(LIVE,2400)
        run('delete job '+LIVE+' --cascade=foreground --wait=true --timeout=55s')
        stage('exporting');manifest('swm-confirm-reader.json');run('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
        try:
            names=['smoke-audit.json','development/training-21.json']+['evaluation/'+n for n in ('audit.json','training-diagnosis.json','summary.json','predictions.json','test.jsonl','train-features.npz','test-features.npz','probes.npz','fitted.json')]
            for name in names:
                # Binary arrays need an unmodified byte stream.
                cmd=K+' exec swm-confirm-reader -- cat /results/micro-auxiliary-v1/'+name
                r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo',HOST,cmd],capture_output=True,timeout=60)
                if r.returncode:raise RuntimeError(r.stderr.decode())
                (OUT/Path(name).name).write_bytes(r.stdout)
        finally:run('delete pod swm-confirm-reader --grace-period=1')
        stage('evaluated')
    except Exception as e:
        connectivity_lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired))
        stage('monitoring_unavailable' if connectivity_lost else 'failed',error=str(e));raise
    finally:
        if connectivity_lost:
            print('Remote state unknown: preserving bounded jobs; reconnect and restore namespace.',flush=True)
        if not connectivity_lost:
            for name in (TRAIN,LIVE):run('delete job '+name+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=1');ready()
            (OUT/'restored.json').write_text(json.dumps({'source':'microservice-source-v10','all_deployments_ready':True},indent=2))
            print('BASELINE_RESTORED',flush=True)
