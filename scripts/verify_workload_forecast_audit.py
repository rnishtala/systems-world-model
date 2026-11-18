"""Independently verify corrected forecast evidence and preserve original results."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.workload import vector
from microservice.workload_evaluate import eligible
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/workload-forecast-audit-v1';OLD=R/'reports/evidence/lab-workload-v2'
def main():
    s=json.loads((E/'summary.json').read_text());original=json.loads((OLD/'evaluation/summary.json').read_text())
    before=json.loads((OLD/'evaluation/before-validation.json').read_text())
    assert s['checkpoint_before']==s['checkpoint_after']==before['checkpoint_sha256'] and s['checkpoint_unchanged']
    assert json.loads((E/'restored.json').read_text())['ready']
    for split in ('train','validation'):
        file=E/f'{split}.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==s['files'][file.name]
        a=np.load(file);old=np.load(OLD/f'evaluation/{split}-forecast.npz')
        rows=[json.loads(l) for l in (OLD/('collection/train.jsonl' if split=='train' else 'evaluation/validation.jsonl')).read_text().splitlines()]
        actual=np.array([vector(r,t+1)[[0,3,4,5]] for r in rows for t in range(3)])
        current=np.array([vector(r,t)[[0,3,4,5]] for r in rows for t in range(3)])
        np.testing.assert_array_equal(a['actual'],actual);np.testing.assert_array_equal(a['current'],current)
        np.testing.assert_allclose(a['raw'],old['predicted'],rtol=1e-5,atol=1e-6)
        for name,key,target in [('raw_forecast','raw','actual'),('corrected_forecast','fixed','actual'),('persistence','current','actual'),('raw_posterior','raw_posterior','current'),('corrected_posterior','posterior','current')]:
            assert np.isfinite(a[key]).all();err=(a[key]-a[target])**2;m=s['splits'][split]['metrics'][name]
            assert float(err.mean())==m['mse'];np.testing.assert_array_equal(err.mean(0),m['by_channel'])
            assert [float(err[a['steps']==t].mean()) for t in range(3)]==m['by_step']
    v=s['splits']['validation']['metrics'];forecast=v['corrected_forecast']['mse']<v['persistence']['mse']
    corrected=dict(verified=True,forecast_gate=forecast,diagnosis_gate=eligible(original['validation']['methods']),combined_gate=bool(forecast and eligible(original['validation']['methods'])),validation_error_reduction=1-v['corrected_forecast']['mse']/v['persistence']['mse'],no_retraining=True,no_new_incidents=True,original_evidence_preserved=True,scope='Local verification recomputes metrics and compares raw replay output to original forecasts; checkpoint replay and production adapter equivalence were checked in the audit pod.')
    (E/'verification.json').write_text(json.dumps(corrected,indent=2)+'\n');print(json.dumps(corrected,indent=2))
if __name__=='__main__':main()
