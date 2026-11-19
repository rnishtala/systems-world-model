"""One operational replacement test; unchanged frozen diagnostic readouts."""
import hashlib,json,subprocess,sys,time
from scripts.run_ledger_probe import R,O,run
NAME='swm-ledger-probe-v2';CM='swm-ledger-probe-code-v2'

def main():
    assert not (O/'test.jsonl').exists()
    frozen=json.loads((O/'frozen.json').read_text())
    for p,h in frozen['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    assert not any(j.get('status',{}).get('active',0) for j in json.loads(run('get jobs -o json'))['items'])
    before=json.loads(run('get deployment swm-inference -o json'))['spec']
    assert before==json.loads((O/'before.json').read_text())
    source=(R/'microservice/ledger_probe.py').read_text()
    replacement=source.replace('else 9102;', 'else 9103;');assert replacement!=source
    code={'live.py':(R/'microservice/live.py').read_text(),'ledger_probe.py':replacement}
    (O/'recovery-collector.py').write_text(replacement)
    (O/'recovery.json').write_text(json.dumps(dict(utc=time.time(),original_seed=9102,replacement_seed=9103,
        collector_hashes={n:hashlib.sha256(v.encode()).hexdigest() for n,v in code.items()},
        readouts_sha256=hashlib.sha256((O/'readouts.npz').read_bytes()).hexdigest(),
        protocol_amendment_sha256=hashlib.sha256((R/'reports/LEDGER_PROBE_RECOVERY.md').read_bytes()).hexdigest()),indent=2)+'\n')
    cm=dict(apiVersion='v1',kind='ConfigMap',metadata=dict(name=CM,namespace='systems-world-model-lab'),data=code)
    manifest=json.loads((R/'deploy/ledger-probe-v1.json').read_text());manifest['metadata']['name']=NAME
    pod=manifest['spec']['template']['spec'];pod['containers'][0]['command'][-1]='test';pod['volumes'][0]['configMap']['name']=CM
    paused=False;created=False;monitor_errors=[]
    try:
        paused=True;print(run('scale deployment swm-inference --replicas=0'),flush=True)
        for _ in range(30):
            cpu=json.loads(run('get resourcequota lab-budget -o json'))['status']['used']['limits.cpu']
            value=float(cpu[:-1])/1000 if cpu.endswith('m') else float(cpu)
            if value<=3.6:break
            time.sleep(2)
        else:raise RuntimeError('Quota not released')
        print(run('create -f -',json.dumps(cm)),flush=True)
        print(run('create -f -',json.dumps(manifest)),flush=True);created=True;deadline=time.monotonic()+1500
        while time.monotonic()<deadline:
            try:
                status=json.loads(run('get job '+NAME+' -o json'))['status']
                log=run('logs job/'+NAME);(O/'test-partial.log').write_text(log)
                print('TEST_PROGRESS',sum(l.startswith('ROW ') for l in log.splitlines()),flush=True)
                if status.get('failed') or status.get('succeeded'):
                    # Fetch again after terminal status so the final reset/completion is included.
                    log=run('logs job/'+NAME);(O/'test.log').write_text(log)
                    (O/'test-job.json').write_text(json.dumps(status,indent=2))
                    lines=[l[4:] for l in log.splitlines() if l.startswith('ROW ')]
                    (O/'test.jsonl').write_text('\n'.join(lines)+'\n')
                    assert status.get('succeeded') and len(lines)==32 and 'RESET_COMPLETE' in log and 'COMPLETE test' in log
                    break
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                monitor_errors.append(dict(utc=time.time(),error=str(exc)[-2000:]))
                (O/'monitor-errors.json').write_text(json.dumps(monitor_errors,indent=2)+'\n')
                print('MONITOR_RETRY_JOB_NOT_RESTARTED',flush=True)
            time.sleep(15)
        else:raise TimeoutError('Recovery monitor deadline; retain partial log')
    finally:
        try:
            if created:print(run('delete job '+NAME+' --ignore-not-found=true --wait=true --timeout=55s'),flush=True)
            print(run('delete configmap '+CM+' --ignore-not-found=true'),flush=True)
        finally:
            if paused:
                print(run('scale deployment swm-inference --replicas='+str(before['replicas'])),flush=True)
                print(run('rollout status deployment/swm-inference --timeout=55s'),flush=True)
                after=json.loads(run('get deployment swm-inference -o json'))['spec'];assert after==before
                (O/'restored.json').write_text(json.dumps(dict(template_unchanged=True,replicas=after['replicas'],recovery=True)))
    subprocess.run([sys.executable,str(R/'scripts/evaluate_ledger_probe.py'),'evaluate'],check=True,cwd=R)
if __name__=='__main__':main()
