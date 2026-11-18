"""Read-only final health check confined to the experiment namespace."""
import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'reports/evidence/lab-workload-v2'
def main():
    command='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab get deployment,pod,job -o json'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo',command],capture_output=True,text=True,check=True,timeout=30)
    items=json.loads(p.stdout)['items'];before=json.loads((R/'before.json').read_text())
    dep=next(i for i in items if i['kind']=='Deployment' and i['metadata']['name']=='swm-inference')
    assert dep['spec']['template']==before['template']
    assert dep['spec']['replicas']==before['replicas']==dep['status']['availableReplicas']==1
    pods=[i for i in items if i['kind']=='Pod'];jobs=[i for i in items if i['kind']=='Job']
    assert not any(i['status'].get('active',0) for i in jobs)
    assert not any(any(s in i['metadata']['name'] for s in ('workload','stage-v1')) for i in pods)
    assert all(i['status']['phase']=='Running' and all(c['ready'] for c in i['status']['containerStatuses']) for i in pods)
    data=dict(namespace='systems-world-model-lab',inference_template_unchanged=True,inference_replicas=1,active_jobs=0,all_remaining_pods_ready=True,pods=[i['metadata']['name'] for i in pods])
    (R/'final-health.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
if __name__=='__main__':main()
