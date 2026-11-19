"""Stage only this experiment, tolerate monitor failures, restore and export."""
import hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path
from scripts.run_ledger_probe import run,SSH,K
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-live-budget-v1'
NAME='swm-probe-live-budget-v1';REMOTE='/results/probe-live-budget-v1';STAGE='swm-probe-live-budget-stage-v1'

def main():
    assert not any(j.get('status',{}).get('active',0) for j in json.loads(run('get jobs -o json'))['items'])
    assert not (O/'before.json').exists()
    source=[]
    for folder in ('microservice','experiment','scripts'):
        source += sorted((R/folder).rglob('*.py'))
    source += [R/'reports/PROBE_LIVE_BUDGET_PROTOCOL.md',R/'deploy/probe-live-budget-v1.json']
    hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    (O/'source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=Path('/tmp/swm-probe-live-budget-v1.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in source:tar.add(path,arcname=str(path.relative_to(R)))

        for path in sorted((O/'frozen').iterdir()):tar.add(path,arcname='frozen/'+path.name)
        tar.add(O/'train.jsonl',arcname='collection/train.jsonl');tar.add(O/'source-hashes.json',arcname='collection/source-hashes.json')
    stage=json.loads((R/'deploy/rcaeval-stage-v1.json').read_text());stage['metadata']['name']=STAGE
    stage['spec']['containers'][0]['command']=['sleep','1800']
    print(run('create -f -',json.dumps(stage)),flush=True)
    try:
        run('wait pod/'+STAGE+' --for=condition=Ready --timeout=55s')
        run('exec '+STAGE+' -- mkdir '+REMOTE)
        with archive.open('rb') as f:subprocess.run(SSH+[K+'exec -i '+STAGE+' -- tar xzf - -C '+REMOTE],stdin=f,check=True,timeout=90)
    finally:print(run('delete pod '+STAGE+' --wait=true --timeout=55s --ignore-not-found=true'),flush=True)
    before=json.loads(run('get deployment swm-inference -o json'))['spec'];(O/'before.json').write_text(json.dumps(before,indent=2))
    paused=False;created=False;failed=False
    try:
        paused=True;print(run('scale deployment swm-inference --replicas=0'),flush=True)
        for _ in range(30):
            cpu=json.loads(run('get resourcequota lab-budget -o json'))['status']['used']['limits.cpu']
            value=float(cpu[:-1])/1000 if cpu.endswith('m') else float(cpu)
            if value<=2:break
            time.sleep(2)
        else:raise RuntimeError('Quota not released')
        manifest=(R/'deploy/probe-live-budget-v1.json').read_text();print(run('create -f -',manifest),flush=True);created=True
        deadline=time.monotonic()+3900;seen=set()
        while time.monotonic()<deadline:
            try:
                status=json.loads(run('get job '+NAME+' -o json'))['status']
                log=run('logs job/'+NAME);(O/'run.log').write_text(log)
                for line in log.splitlines():
                    if line.startswith(('PROBE_','TRAINING_COMPLETE')) and line not in seen:
                        print(line[:1000],flush=True);seen.add(line)
                if status.get('failed') or status.get('succeeded'):
                    (O/'job.json').write_text(json.dumps(status,indent=2));(O/'run.log').write_text(run('logs job/'+NAME));failed=bool(status.get('failed'));break
                print('EXPERIMENT_ACTIVE',flush=True)
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                with (O/'monitor-errors.jsonl').open('a') as f:f.write(json.dumps(dict(utc=time.time(),error=str(exc)[-1500:]))+'\n')
                print('MONITOR_RETRY_JOB_NOT_RESTARTED',flush=True)
            time.sleep(20)
        else:raise TimeoutError('Bounded experiment deadline')
    finally:
        try:
            if created:print(run('delete job '+NAME+' --ignore-not-found=true --wait=true --timeout=55s'),flush=True)
        finally:
            if paused:
                print(run('scale deployment swm-inference --replicas='+str(before['replicas'])),flush=True)
                print(run('rollout status deployment/swm-inference --timeout=55s'),flush=True)
                after=json.loads(run('get deployment swm-inference -o json'))['spec'];assert after==before
                (O/'restored.json').write_text(json.dumps(dict(template_unchanged=True,replicas=after['replicas'])))
    subprocess.run([sys.executable,str(R/'scripts/export_probe_live_budget.py')],check=True,cwd=R)
    if failed:raise RuntimeError('Experiment failed; inspect saved evidence')
if __name__=='__main__':main()
