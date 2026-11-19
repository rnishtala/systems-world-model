"""Namespace-only pilot lifecycle, local evidence and guaranteed restoration attempt."""
import json,subprocess,time,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/ledger-probe-v1'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo']
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab '
NAME='swm-ledger-probe-v1';CM='swm-ledger-probe-code-v1'

def run(cmd,body=None):
    for attempt in range(5):
        try:
            p=subprocess.run(SSH+[K+cmd],input=body,capture_output=True,text=True,timeout=65)
            if p.returncode:raise RuntimeError(p.stderr[-2000:])
            return p.stdout
        except (RuntimeError,subprocess.TimeoutExpired):
            if body is not None or attempt==4:raise
            time.sleep(2)

def main():
    O.mkdir(exist_ok=False)
    assert not any(j.get('status',{}).get('active',0) for j in json.loads(run('get jobs -o json'))['items'])
    before=json.loads(run('get deployment swm-inference -o json'))['spec'];(O/'before.json').write_text(json.dumps(before,indent=2))
    code={n:(R/'microservice'/n).read_text() for n in ('live.py','ledger_probe.py')}
    cm=dict(apiVersion='v1',kind='ConfigMap',metadata=dict(name=CM,namespace='systems-world-model-lab'),data=code)
    manifest=json.loads((R/'deploy/ledger-probe-v1.json').read_text())
    (O/'collector-hashes.json').write_text(json.dumps({n:hashlib.sha256(v.encode()).hexdigest() for n,v in code.items()},indent=2))
    paused=False
    try:
        paused=True;print(run('scale deployment swm-inference --replicas=0'),flush=True)
        print(run('wait --for=delete pod -l app=swm-inference --timeout=55s'),flush=True)
        # Explicitly wait for quota, independent of deployment label conventions.
        for _ in range(30):
            cpu=json.loads(run('get resourcequota lab-budget -o json'))['status']['used']['limits.cpu']
            value=float(cpu[:-1])/1000 if cpu.endswith('m') else float(cpu)
            if value<=3.6:break
            time.sleep(2)
        else:raise RuntimeError('Quota not released')
        print(run('create -f -',json.dumps(cm)),flush=True)
        for split in ('development','test'):
            manifest['spec']['template']['spec']['containers'][0]['command'][-1]=split
            print(run('create -f -',json.dumps(manifest)),flush=True)
            for _ in range(80):
                status=json.loads(run('get job '+NAME+' -o json'))['status']
                if status.get('failed') or status.get('succeeded'):break
                print('COLLECTING',split,flush=True);time.sleep(15)
            else:raise RuntimeError('Collection deadline exceeded')
            log=run('logs job/'+NAME);(O/f'{split}.log').write_text(log)
            (O/f'{split}-job.json').write_text(json.dumps(status,indent=2))
            lines=[line[4:] for line in log.splitlines() if line.startswith('ROW ')]
            (O/f'{split}.jsonl').write_text('\n'.join(lines)+'\n')
            assert status.get('succeeded') and len(lines)==32 and 'RESET_COMPLETE' in log and 'COMPLETE '+split in log
            print(run('delete job '+NAME+' --wait=true --timeout=55s'),flush=True)
            if split=='development':subprocess.run([sys.executable,str(R/'scripts/evaluate_ledger_probe.py'),'freeze'],check=True,cwd=R)
    finally:
        try:
            print(run('delete job '+NAME+' --ignore-not-found=true --wait=true --timeout=55s'),flush=True)
            print(run('delete configmap '+CM+' --ignore-not-found=true'),flush=True)
        finally:
            if paused:
                print(run('scale deployment swm-inference --replicas='+str(before['replicas'])),flush=True)
                print(run('rollout status deployment/swm-inference --timeout=55s'),flush=True)
                after=json.loads(run('get deployment swm-inference -o json'))['spec']
                assert after['template']==before['template'] and after['replicas']==before['replicas']
                (O/'restored.json').write_text(json.dumps(dict(template_unchanged=True,replicas=after['replicas'])))
    subprocess.run([sys.executable,str(R/'scripts/evaluate_ledger_probe.py'),'evaluate'],check=True,cwd=R)
if __name__=='__main__':main()
