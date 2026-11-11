"""Separate reviewed promotion after the experiment's baseline-restoration drill."""
import json,subprocess,time
from pathlib import Path
from scripts import finish_window as ops
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/evidence/latent-api'

def main():
    ops.OUT=OUT;summary=json.loads((OUT/'summary.json').read_text());verified=json.loads((OUT/'verified.json').read_text());original=json.loads((OUT/'original-sources.json').read_text())
    assert summary['parity_passed'] and summary['gate_passed'] and verified['baseline_restored'] and verified['parity_matching']==80
    assert not (OUT/'promotion.json').exists(),'Preserve prior promotion evidence'
    lost=False
    try:
        ops.manifest('microservice-agent-latent-api.json','apply')
        for name in original:ops.patch_source(name,'microservice-source-v20')
        ops.ready();ops.manifest('swm-latent-api-smoke-v1.json');ops.wait_job('swm-latent-api-smoke-v1',600)
        ops.run('delete job swm-latent-api-smoke-v1 --cascade=foreground --wait=true --timeout=55s')
        ops.export_files([('promotion-smoke.json','promotion-smoke.json')],remote_root='/results/micro-latent-api-v1/')
        smoke=json.loads((OUT/'promotion-smoke.json').read_text());assert smoke['healthy'] and smoke['valid_windows']
        deployments=json.loads(ops.run('get deployment swm-checkout swm-payment swm-ledger swm-postgres swm-inference -o json'))['items'];state=[]
        for d in deployments:
            name=d['metadata']['name'];status=d['status'];assert d['spec']['replicas']==status.get('availableReplicas')==status.get('updatedReplicas')==1
            source=next((v['configMap']['name'] for v in d['spec']['template']['spec']['volumes'] if v['name']=='source'),None)
            if name!='swm-postgres':assert source=='microservice-source-v20'
            state.append(dict(name=name,ready=True,source=source))
        result=dict(promoted=True,source='microservice-source-v20',model_seed=23,readout_sha256=smoke['readout_sha256'],healthy_smoke_passed=True,deployments=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='Separate reviewed lab-only promotion after experiment restoration; diagnosis only.')
        (OUT/'promotion.json').write_text(json.dumps(result,indent=2)+'\n')
        (ROOT/'deploy/microservice-agent.json').write_bytes((ROOT/'deploy/microservice-agent-latent-api.json').read_bytes())
        print('PROMOTED',json.dumps(result),flush=True)
    except Exception as e:
        lost=isinstance(e,(ConnectionError,subprocess.TimeoutExpired))
        (OUT/'promotion-error.json').write_text(json.dumps(dict(error=str(e),connectivity_lost=lost),indent=2)+'\n')
        if not lost:
            ops.run('delete job swm-latent-api-smoke-v1 --ignore-not-found --cascade=foreground --wait=true --timeout=55s')
            ops.manifest('microservice-agent-api-restore.json','apply')
            for name,source in original.items():ops.patch_source(name,source)
            ops.ready()
        raise
if __name__=='__main__':main()
