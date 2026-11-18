"""Fixed development-only diagnostic; never reads confirmation observations."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from microservice.order_compare import KEYS, choose_order
from microservice.partial_readout import outer_fit, folds, score, assess
from microservice.efficiency import decisions
from microservice.mechanisms import FAULT_MASK

ROOT = Path(__file__).resolve().parents[1]

def main():
    paths = {
        'development': ROOT/'reports/evidence/lab-efficiency-v1/evaluation/development.jsonl',
        'features': ROOT/'reports/evidence/lab-order-v1/evaluation/features.npz',
        'policy': ROOT/'reports/evidence/lab-order-confirm-v1/frozen/policy.json',
        'readouts': ROOT/'reports/evidence/lab-order-confirm-v1/frozen/readouts.npz',
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    policy = json.loads(paths['policy'].read_text())
    assert hashes['development'] == policy['development_sha256']
    assert hashes['features'] == policy['features_sha256']
    assert hashes['readouts'] == policy['readouts_sha256']
    rows = [json.loads(line) for line in paths['development'].read_text().splitlines()]
    assert len(rows) == 80 and all(r['seed'] == 8201 for r in rows)
    families = np.array([r['fault'] for r in rows])
    target = np.array([FAULT_MASK[f] for f in families])
    y = ((target[:,None] >> np.arange(3)) & 1)*2-1
    fs = np.load(paths['features']); params = np.load(paths['readouts'])
    result = {'scope': 'Development diagnostics only; no independent performance estimate', 'hashes': hashes, 'methods': {}}
    for name in ('latent','raw'):
        records = []; transfer = {}
        for seed in (8400, *range(8500,8510)):
            inner = folds(families,5,seed)
            runs = {key: outer_fit([fs[f'{name}_{key}_{t}'] for t in range(4)],y,families,np.arange(80),np.array([],int),inner) for key in KEYS}
            order, threshold, _ = choose_order(runs,target,families)
            run = runs[order]
            records.append(dict(seed=seed,order=order,threshold=threshold,readouts=run['selected'],selection_stats=assess(*decisions(run['inner_scores'],threshold),target,families)))
            if seed == 8400:
                frozen = policy['policies'][name]
                assert order == frozen['order'] and threshold == frozen['threshold'] and run['selected'] == frozen['readouts']
                fitted = []
                for t,config in enumerate(frozen['readouts']):
                    model = dict(config=config,**{k:params[f'{name}_{t}_{k}'] for k in ('mean','scale','bias','z','dual')})
                    fitted.append(score(model,fs[f'{name}_{order}_{t}']))
                for label, values in [('selected_out_of_fold',run['inner_scores']),('final_fit_in_sample',np.stack(fitted,axis=1))]:
                    margins = np.min(np.abs(values),axis=2)
                    transfer[label] = dict(stats=assess(*decisions(values,threshold),target,families),margin_quantiles=np.quantile(margins,[0,.25,.5,.75,1],axis=0).tolist(),above_threshold_per_prefix=(margins>=threshold).sum(0).tolist() if threshold is not None else None)
            print(name,seed,order,threshold,flush=True)
        result['methods'][name] = dict(transfer=transfer,selections=records,order_counts=dict(Counter(r['order'] for r in records[1:])))
    assert hashes == {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    out = ROOT/'reports/evidence/lab-order-stability-v1';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
