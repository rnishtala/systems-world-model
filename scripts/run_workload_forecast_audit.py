"""Run checkpoint audit with read-only PVC; restore inference and remove audit pod."""
import base64,hashlib,json,shlex,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R/'reports/evidence/workload-forecast-audit-v1';OUT.mkdir(exist_ok=False)
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo'];K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab ';POD='swm-forecast-audit-v1'
def run(cmd,data=None):
    for attempt in range(5):
        try:
            p=subprocess.run(SSH+[K+cmd],input=data,capture_output=True,text=True,timeout=60)
            if p.returncode:raise RuntimeError(p.stderr[-1800:])
            return p.stdout
        except (RuntimeError,subprocess.TimeoutExpired):
            if attempt==4:raise
            time.sleep(2)
def main():
    assert not any(j['status'].get('active',0) for j in json.loads(run('get jobs -o json'))['items'])
    before=json.loads(run('get deployment swm-inference -o json'))['spec'];(OUT/'before.json').write_text(json.dumps(before,indent=2)+'\n')
    pod=json.loads((R/'deploy/rcaeval-stage-v1.json').read_text());pod['metadata']['name']=POD
    c=pod['spec']['containers'][0];c['resources']={'requests':{'cpu':'1','memory':'3Gi'},'limits':{'cpu':'2','memory':'4Gi'}}
    c['volumeMounts'][0]['readOnly']=True
    files={'audit.py':R/'scripts/audit_workload_forecast.py','microservice/agent.py':R/'microservice/agent.py','microservice/decoder_units.py':R/'microservice/decoder_units.py'}
    payload={n:base64.b64encode(p.read_bytes()).decode() for n,p in files.items()}
    unpack="import base64,json,pathlib; data=json.loads("+repr(json.dumps(payload))+"); [(pathlib.Path('/tmp')/n).write_bytes(base64.b64decode(v)) for n,v in data.items()]"
    c['command']=['sh','-ec','cp -r /results/lab-workload-v2/microservice /tmp/microservice\n/results/venv/bin/python -c '+shlex.quote(unpack)+'\nexport PYTHONPATH=/tmp:/results/lab-workload-v2:/results/work/upstream\nexport DREAMER_SOURCE=/results/work/upstream\n/results/venv/bin/python /tmp/audit.py\ntouch /tmp/audit-done\nsleep 1800']
    (OUT/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()},indent=2)+'\n')
    paused=False;created=False
    try:
        run('scale deployment swm-inference --replicas=0');paused=True
        run('wait --for=delete pod -l swm-component=inference --timeout=55s')
        run('create -f -',json.dumps(pod));created=True
        run('wait pod/'+POD+' --for=condition=Ready --timeout=55s')
        for _ in range(60):
            p=json.loads(run('get pod '+POD+' -o json'))
            if p['status']['phase'] in ('Failed','Succeeded'):raise RuntimeError('Audit exited before export: '+p['status']['phase'])
            done=run('exec '+POD+' -- sh -c '+shlex.quote('test ! -f /tmp/audit-done || echo done')).strip()
            if done:break
            print('AUDIT_ACTIVE',flush=True);time.sleep(10)
        else:raise TimeoutError('Audit deadline')
        for name in ('summary.json','train.npz','validation.npz'):
            remote='/tmp/forecast-audit/'+name
            size=int(run('exec '+POD+' -- stat -c %s '+remote).strip());data=bytearray()
            for offset in range(0,size,32768):
                command=f'dd if={remote} bs=32768 skip={offset//32768} count=1 status=none | base64 -w0'
                data.extend(base64.b64decode(run('exec '+POD+' -- sh -c '+shlex.quote(command)).strip(),validate=True))
            assert len(data)==size;(OUT/name).write_bytes(data)
        summary=json.loads((OUT/'summary.json').read_text())
        for name,h in summary['files'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
        print('AUDIT_EXPORTED',flush=True)
    finally:
        if created:
            try:(OUT/'run.log').write_text(run('logs '+POD))
            finally:run('delete pod '+POD+' --grace-period=1 --wait=true --timeout=55s --ignore-not-found=true')
        if paused:
            run('scale deployment swm-inference --replicas='+str(before['replicas']))
            run('rollout status deployment/swm-inference --timeout=55s')
            after=json.loads(run('get deployment swm-inference -o json'));assert before['template']==after['spec']['template']
            assert after['status']['availableReplicas']==before['replicas']
            (OUT/'restored.json').write_text(json.dumps(dict(template_unchanged=True,replicas=before['replicas'],ready=True))+'\n')
if __name__=='__main__':main()
