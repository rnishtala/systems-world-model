"""Run the frozen development experiment and always restore the baseline.
Namespace is fixed. No retry/reuse of failed experiments or alteration of gates.
"""
import hashlib,json,subprocess,time,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/auxiliary';OUT.mkdir(parents=True,exist_ok=True)
HOST='rajnishtala@amd-halo';NS='systems-world-model-lab'
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+NS
TRAIN='swm-auxiliary-train-v2';LIVE='swm-auxiliary-eval-v2'
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
        stage('restarting_incomplete_evaluation',scenario_seed=6501)
        run('create configmap microservice-source-v17 --from-file=app.tar.gz=/tmp/swm-micro-v17.tar.gz')
        run('scale deployment swm-inference --replicas=0')
        run('wait --for=delete pod -l swm-component=inference --timeout=55s')
        run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1')
        run('create -f /tmp/swm-auxiliary-eval-v2.json')
        stage('evaluating',scenario_seed=6501);wait_job(LIVE,2400)
        run('delete job '+LIVE+' --cascade=foreground --wait=true --timeout=55s')
        stage('exporting');manifest('swm-confirm-reader.json');run('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
        try:
            names=[('micro-auxiliary-v1/smoke-audit.json','smoke-audit.json'),('micro-auxiliary-v1/development/training-21.json','training-21.json'),('micro-auxiliary-v1/evaluation/test.jsonl','incomplete-6500.jsonl')]+[('micro-auxiliary-v2/evaluation/'+n,n) for n in ('audit.json','training-diagnosis.json','summary.json','predictions.json','test.jsonl','train-features.npz','test-features.npz','probes.npz','fitted.json')]
            for remote_name,local_name in names:
                cmd=K+' exec swm-confirm-reader -- cat /results/'+remote_name
                for attempt in range(10):
                    try:r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo',HOST,cmd],capture_output=True,timeout=60)
                    except subprocess.TimeoutExpired:continue
                    if r.returncode==0:break
                    if r.returncode!=255:raise RuntimeError(r.stderr.decode())
                else:raise ConnectionError('Export retries exhausted')
                (OUT/local_name).write_bytes(r.stdout)
        finally:run('delete pod swm-confirm-reader --grace-period=1')
        stage('evaluated',scenario_seed=6501)
    except Exception as e:
        connectivity_lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired))
        stage('monitoring_unavailable' if connectivity_lost else 'failed',error=str(e));raise
    finally:
        if connectivity_lost:print('Remote state unknown: preserving bounded job; reconnect and restore namespace.',flush=True)
        else:
            run('delete job '+LIVE+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=1');ready()
            (OUT/'restored.json').write_text(json.dumps({'source':'microservice-source-v10','all_deployments_ready':True},indent=2))
            print('BASELINE_RESTORED',flush=True)
