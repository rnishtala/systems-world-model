"""Archive terminal jobs; optionally delete them, only in the world-model lab."""
import argparse,concurrent.futures,datetime,hashlib,json,subprocess
from pathlib import Path

NS='systems-world-model-lab'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo','rajnishtala@amd-halo']
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n '+NS+' '

def run(command):
    result=subprocess.run(SSH+[K+command],capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return result.stdout

def terminal(job):
    status=job.get('status',{})
    return not status.get('active',0) and any(c['type'] in ('Complete','Failed') and c['status']=='True' for c in status.get('conditions',[]))

def snapshot():
    deployments=json.loads(run('get deployments -o json'))['items']
    pvcs=json.loads(run('get pvc -o json'))['items']
    return dict(deployments={d['metadata']['name']:d['spec'] for d in deployments},pvcs={p['metadata']['name']:p['metadata']['uid'] for p in pvcs})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts'/('job-cleanup-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
    before=snapshot();(out/'before.json').write_text(json.dumps(before,indent=2))
    jobs=[j for j in json.loads(run('get jobs -o json'))['items'] if terminal(j)]
    pods=json.loads(run('get pods -o json'))['items']
    uids={j['metadata']['uid'] for j in jobs}
    targets=[p for p in pods if any(o['kind']=='Job' and o['uid'] in uids for o in p['metadata'].get('ownerReferences',[]))]
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2));(out/'pods.json').write_text(json.dumps(targets,indent=2))
    def archive(pod):
        name=pod['metadata']['name']
        text=run('logs '+name+' --all-containers=true --timestamps=true')
        (out/(name+'.log')).write_text(text)
    # Finish all archives successfully before any deletion.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(archive,targets))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2));print('ARCHIVED',len(jobs),'jobs;',len(targets),'pod logs;',out,flush=True)
    deleted=[]
    if args.apply:
        for job in jobs:
            name=job['metadata']['name'];current=json.loads(run('get job '+name+' -o json'))
            assert current['metadata']['uid']==job['metadata']['uid'] and terminal(current),name
            run('delete job '+name+' --cascade=foreground --wait=true --timeout=45s');deleted.append(name)
    after=snapshot();assert before==after,'Deployment specification or PVC identity changed'
    remaining=json.loads(run('get pods -o json'))['items']
    summary=dict(namespace=NS,archived_jobs=len(jobs),archived_pod_logs=len(targets),deleted_jobs=deleted,archive=str(out),deployments_and_pvcs_unchanged=True,remaining_pods=[dict(name=p['metadata']['name'],phase=p['status']['phase']) for p in remaining])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
