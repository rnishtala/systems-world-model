"""Verify objective selection against actual training metrics and scalar gradients."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    meta=json.loads((root/'metadata.json').read_text());control=meta['auxiliary_only']
    rows=[json.loads(line) for line in (root/'train/metrics.jsonl').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if 'train/optimized_loss' in r and 'train/diagnosis_loss' in r]
    assert rows,'No actual training-objective metrics'
    for r in rows:
        assert np.isfinite(r['train/optimized_loss']) and np.isfinite(r['train/diagnosis_loss'])
        assert r['train/world_model_objective_enabled']==float(not control)
        if control:assert np.isclose(r['train/optimized_loss'],r['train/diagnosis_loss'],rtol=1e-6,atol=1e-7)
    import jax
    from benchmark.supervision import objective
    derivative=float(jax.grad(lambda base:objective(base,2.,control))(7.))
    assert derivative==float(not control)
    result=dict(auxiliary_only=control,training_metric_rows_checked=len(rows),optimized_equals_diagnosis=all(np.isclose(r['train/optimized_loss'],r['train/diagnosis_loss'],rtol=1e-6,atol=1e-7) for r in rows),base_loss_scalar_derivative=derivative,scope='Actual optimized-loss metrics plus derivative of the loss-combination function; not a per-parameter gradient audit.')
    (root/'objective-audit.json').write_text(json.dumps(result,indent=2));print('OBJECTIVE_AUDIT',json.dumps(result),flush=True)
if __name__=='__main__':main()
