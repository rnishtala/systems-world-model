"""Bounded API integration experiment; restore exact original inference spec."""
import base64,hashlib,json,subprocess,time
from pathlib import Path
from scripts import finish_window as ops
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/latent-api'
SOURCE='microservice-source-v20';JOBS=['swm-latent-api-package-v1','swm-latent-api-live-v1']

def main():
    OUT.mkdir(parents=True,exist_ok=False);ops.OUT=OUT;started=False;connectivity_lost=False;original={}
    try:
        active=[j['metadata']['name'] for j in json.loads(ops.run('get jobs -o json'))['items'] if j.get('status',{}).get('active',0)]
        if active:raise RuntimeError('Lab active jobs: '+str(active))
        ops.ready();deployments=json.loads(ops.run('get deployment swm-checkout swm-payment swm-ledger swm-inference -o json'))['items']
        for d in deployments:
            name=d['metadata']['name']
            if name=='swm-inference':
                baseline={'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':name,'namespace':ops.NS},'spec':d['spec']}
                (ROOT/'deploy/microservice-agent-api-restore.json').write_text(json.dumps(baseline,indent=2)+'\n')
            else:original[name]=next(v['configMap']['name'] for v in d['spec']['template']['spec']['volumes'] if v['name']=='source')
        (OUT/'original-sources.json').write_text(json.dumps(original,indent=2)+'\n')
        source=json.loads(Path('/tmp/swm-micro-v20-source.json').read_text());source.update(protocol_sha256=hashlib.sha256((ROOT/'reports/LATENT_API_PROTOCOL.md').read_bytes()).hexdigest(),recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (OUT/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        subprocess.run(['scp','-o','BatchMode=yes','-o','HostName=100.98.114.55','-o','HostKeyAlias=amd-halo','/tmp/swm-micro-v20.tar.gz',ops.HOST+':/tmp/swm-micro-v20.tar.gz'],check=True,timeout=45)
        ops.run('create configmap '+SOURCE+' --from-file=app.tar.gz=/tmp/swm-micro-v20.tar.gz');started=True
        ops.stage('packaging');ops.manifest(JOBS[0]+'.json');ops.wait_job(JOBS[0],600)
        ops.run('delete job '+JOBS[0]+' --cascade=foreground --wait=true --timeout=55s')
        ops.stage('deploying_candidate');ops.manifest('microservice-agent-latent-api.json','apply')
        for name in original:ops.patch_source(name,SOURCE)
        ops.ready();ops.stage('api_validation')
        ops.manifest(JOBS[1]+'.json');ops.wait_job(JOBS[1],1800)
        ops.run('delete job '+JOBS[1]+' --cascade=foreground --wait=true --timeout=55s')
        ops.export_files([('package.json','package.json'),('evaluation/parity.json','parity.json'),('evaluation/summary.json','summary.json')],remote_root='/results/micro-latent-api-v1/')
        result=json.loads((OUT/'summary.json').read_text())
        if not result['fresh_skipped']:ops.export_files([('evaluation/live.jsonl','live.jsonl')],remote_root='/results/micro-latent-api-v1/')
        (OUT/'completion.json').write_text(json.dumps(dict(parity_passed=result['parity_passed'],live_gate_passed=result['gate_passed'],promoted=False),indent=2)+'\n');ops.stage('completed',result=result)
    except Exception as e:
        connectivity_lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired));ops.stage('monitoring_unavailable' if connectivity_lost else 'failed',error=str(e));raise
    finally:
        if connectivity_lost:print('Remote state unknown: preserving bounded jobs. Reconnect before cleanup.',flush=True)
        elif started:
            for name in JOBS:ops.run('delete job '+name+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            ops.manifest('microservice-agent-api-restore.json','apply')
            for name,source in original.items():ops.patch_source(name,source)
            ops.run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=1');ops.ready()
            current=json.loads(ops.run('get deployment swm-inference -o json'))
            assert current['spec']['template']==baseline['spec']['template'],'Inference baseline differs'
            (OUT/'restored.json').write_text(json.dumps(dict(original_service_sources=original,inference_template_restored=True,all_deployments_ready=True),indent=2)+'\n');print('BASELINE_RESTORED',flush=True)
if __name__=='__main__':main()
