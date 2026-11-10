"""Execute the preregistered independent latent-readout confirmation only in the lab."""
import base64,hashlib,json,subprocess,time
from pathlib import Path
from scripts import finish_window as ops
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/evidence/latent-confirmation'
JOBS=['swm-latent-confirm-train-23','swm-latent-confirm-eval-23']
SOURCE='microservice-source-v19'

def main():
    OUT.mkdir(parents=True,exist_ok=False);ops.OUT=OUT
    started=False;connectivity_lost=False;original={}
    try:
        active=[j['metadata']['name'] for j in json.loads(ops.run('get jobs -o json'))['items'] if j.get('status',{}).get('active',0)]
        if active:raise RuntimeError('Lab has active jobs: '+str(active))
        ops.ready()
        deployments=json.loads(ops.run('get deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference -o json'))['items']
        for d in deployments:
            name=d['metadata']['name']
            if name in ('swm-checkout','swm-payment','swm-ledger'):
                original[name]=next(v['configMap']['name'] for v in d['spec']['template']['spec']['volumes'] if v['name']=='source')
        (OUT/'original-sources.json').write_text(json.dumps(original,indent=2)+'\n')
        source=json.loads(ops.run('get configmap '+SOURCE+' -o json'))
        sha=hashlib.sha256(base64.b64decode(source['binaryData']['app.tar.gz'])).hexdigest()
        assert sha=='e6a27c2ffda41ae56ffa905a037f1e644e8c14c6700518e6dd4540f39bd450eb'
        provenance=dict(source=SOURCE,source_sha256=sha,protocol_sha256=hashlib.sha256((ROOT/'reports/LATENT_CONFIRMATION_PROTOCOL.md').read_bytes()).hexdigest(),recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),primary='latent',model_seed=23,scenario_seed=6801)
        (OUT/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
        started=True;ops.stage('training',model_seed=23)
        ops.run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=0');ops.drain()
        ops.manifest(JOBS[0]+'.json');ops.wait_job(JOBS[0],2400)
        ops.run('delete job '+JOBS[0]+' --cascade=foreground --wait=true --timeout=55s')
        for name in original:ops.patch_source(name,SOURCE)
        ops.run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres --replicas=1');ops.apps_ready()
        ops.stage('evaluating',model_seed=23,scenario_seed=6801)
        ops.manifest(JOBS[1]+'.json');ops.wait_job(JOBS[1],2400)
        ops.run('delete job '+JOBS[1]+' --cascade=foreground --wait=true --timeout=55s')
        names=['audit.json','training-diagnosis.json','summary.json','predictions.json','test.jsonl','train-features.npz','test-features.npz','probes.npz','fitted.json']
        ops.export_files([(f'eval-23/{n}',f'eval-23/{n}') for n in names]+[(f'models/{n}',n) for n in ('training-23.json','identity-23.json')],remote_root='/results/micro-latent-confirm-v1/')
        result=json.loads((OUT/'eval-23/summary.json').read_text());primary=result['methods']['latent']
        (OUT/'completion.json').write_text(json.dumps(dict(primary='latent',confirmation_passed=primary['gate_passed'],promoted=False),indent=2)+'\n')
        ops.stage('completed',primary=primary)
    except Exception as e:
        connectivity_lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired))
        ops.stage('monitoring_unavailable' if connectivity_lost else 'failed',error=str(e));raise
    finally:
        if connectivity_lost:print('Remote state unknown: preserving bounded jobs. Reconnect before cleanup.',flush=True)
        elif started:
            for name in JOBS:ops.run('delete job '+name+' --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            for name,source in original.items():ops.patch_source(name,source)
            ops.run('scale deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference --replicas=1');ops.ready()
            deployments=json.loads(ops.run('get deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference -o json'))['items']
            for d in deployments:
                if d['metadata']['name'] in original:
                    assert next(v['configMap']['name'] for v in d['spec']['template']['spec']['volumes'] if v['name']=='source')==original[d['metadata']['name']]
            (OUT/'restored.json').write_text(json.dumps(dict(original_service_sources=original,all_deployments_ready=True),indent=2)+'\n')
            print('BASELINE_RESTORED',flush=True)

if __name__=='__main__':main()
