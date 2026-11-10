"""Run the frozen development experiment and always restore the baseline.
Namespace is fixed. No retry/reuse of failed experiments or alteration of gates.
"""
import hashlib,json,subprocess,time,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/measurement-window';OUT.mkdir(parents=True,exist_ok=True)
HOST='rajnishtala@amd-halo';NS='systems-world-model-lab'
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+NS
SOURCE='microservice-source-v19'
JOBS=['swm-window-eval-22','swm-window-train-23','swm-window-eval-23']
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
def export_files(mapping,remote_root='/results/micro-window-v1/'):
    manifest('swm-confirm-reader.json');run('wait pod/swm-confirm-reader --for=condition=Ready --timeout=55s')
    try:
        for remote_name,local_name in mapping:
            cmd=K+' exec swm-confirm-reader -- cat '+remote_root+remote_name
            for attempt in range(20):
                try:r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo',HOST,cmd],capture_output=True,timeout=60)
                except subprocess.TimeoutExpired:continue
                if r.returncode==0:break
                if r.returncode!=255:raise RuntimeError(r.stderr.decode())
            else:raise ConnectionError('Export retries exhausted')
            p=OUT/local_name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.json':json.loads(r.stdout)
            if p.suffix=='.jsonl':
                for line in r.stdout.splitlines():json.loads(line)
            p.write_bytes(r.stdout)
    finally:run('delete pod swm-confirm-reader --grace-period=1')

def drain():
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        pods=json.loads(run('get pods -o json'))['items']
        if not any(p.get('status',{}).get('phase') not in ('Succeeded','Failed') for p in pods):return
        time.sleep(5)
    raise TimeoutError('Lab pods did not drain')

def patch_source(name,source):
    import shlex
    patch={'spec':{'template':{'spec':{'volumes':[{'name':'source','configMap':{'name':source}}]}}}}
    run('patch deployment '+name+' --type=strategic -p '+shlex.quote(json.dumps(patch)))

def apps_ready():
    for name in ('swm-checkout','swm-payment','swm-ledger','swm-postgres'):
        run('rollout status deployment/'+name+' --timeout=55s')

def evaluate(seed):
    name=f'swm-window-eval-{seed}'
    manifest(name+'.json');stage('evaluating',model_seed=seed);wait_job(name,2400)
    run('delete job '+name+' --cascade=foreground --wait=true --timeout=55s')
    names=['audit.json','training-diagnosis.json','summary.json','predictions.json','test.jsonl','train-features.npz','test-features.npz','probes.npz','fitted.json']
    export_files([(f'eval-{seed}/'+n,f'eval-{seed}/'+n) for n in names])
    result=json.loads((OUT/f'eval-{seed}/summary.json').read_text())
    stage('scored',model_seed=seed,primary=result['methods']['dreamer_reward'])
    return result['methods']['dreamer_reward']['gate_passed']

if __name__=='__main__':
    connectivity_lost=False;started=False
    original=json.loads((OUT/'original-sources.json').read_text())
    try:
        active=[j['metadata']['name'] for j in json.loads(run('get jobs -o json'))['items'] if j.get('status',{}).get('active',0)]
        if active:raise RuntimeError('Lab has active jobs: '+str(active))
        ready()
        subprocess.run(['scp','-o','BatchMode=yes','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo','/tmp/swm-micro-v19.tar.gz',HOST+':/tmp/swm-micro-v19.tar.gz'],check=True,timeout=45)
        run('create configmap '+SOURCE+' --from-file=app.tar.gz=/tmp/swm-micro-v19.tar.gz');started=True
        stage('deploying_measurement_fix')
        run('scale deployment swm-inference --replicas=0')
        for name in original:patch_source(name,SOURCE)
        apps_ready()
        development=evaluate(22);confirmation=None
        if development:
            stage('training',model_seed=23)
            run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=0');drain()
            manifest('swm-window-train-23.json');wait_job('swm-window-train-23',2400)
            run('delete job swm-window-train-23 --cascade=foreground --wait=true --timeout=55s')
            run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1');apps_ready()
            confirmation=evaluate(23)
        (OUT/'completion.json').write_text(json.dumps(dict(development_passed=development,confirmation_passed=confirmation,confirmation_skipped=not development,promoted=False),indent=2))
        stage('completed',development_passed=development,confirmation_passed=confirmation)
    except Exception as e:
        connectivity_lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired))
        stage('monitoring_unavailable' if connectivity_lost else 'failed',error=str(e));raise
    finally:
        if connectivity_lost:print('Remote state unknown: preserving bounded jobs. Reconnect before cleanup.',flush=True)
        elif started:
            for name in JOBS:run('delete job '+name+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            for name,source in original.items():patch_source(name,source)
            run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=1');ready()
            (OUT/'restored.json').write_text(json.dumps({'original_service_sources':original,'inference_source':'microservice-source-v10','all_deployments_ready':True},indent=2))
            print('BASELINE_RESTORED',flush=True)
