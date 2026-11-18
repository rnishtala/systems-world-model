"""Prespecified mechanism-specific readout development; no cluster access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from microservice.partial_readout import CANDIDATES, fit, score, diagnose
from microservice.latent_probe import labels
from microservice.workload_evaluate import stats
from scripts.develop_forecast_readout import SOURCE, AUDIT, features, choose

R = Path(__file__).resolve().parents[1]
OUT = R / 'reports/evidence/mechanism-readout-v1'

def select(xs, y, groups, names):
    trials = []
    for name in names:
        for config in CANDIDATES:
            values = np.empty_like(y, dtype=float)
            for group in sorted(set(groups)):
                held = groups == group
                values[held] = score(fit(xs[name][~held], y[~held], config), xs[name][held])
            trials.append(dict(feature=name, config=config,
                bit_correct=np.sum((values > 0) == (y > 0), axis=0).tolist()))
    chosen = [int(np.argmax([t['bit_correct'][bit] for t in trials])) for bit in range(3)]
    return [trials[i] for i in chosen], dict(trials=trials, chosen=chosen)

def predict(xs, y, held_xs, selected):
    result = np.empty((len(next(iter(held_xs.values()))), 3))
    for bit, choice in enumerate(selected):
        name = choice['feature']
        model = fit(xs[name], y[:, bit:bit+1], choice['config'])
        result[:, bit] = score(model, held_xs[name])[:, 0]
    return diagnose(result)

def main():
    OUT.mkdir(exist_ok=False)
    paths = [SOURCE/'collection/train.jsonl', SOURCE/'evaluation/train-features.npz', AUDIT/'train.npz',
             R/'reports/MECHANISM_READOUT_PROTOCOL.md', Path(__file__)]
    hashes = {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = [json.loads(l) for l in paths[0].read_text().splitlines()]
    assert len(rows) == 128 and all(row['seed'] == 8701 for row in rows)
    base = np.load(paths[1]); xs = features(base, np.load(paths[2])); y = labels(rows)
    np.testing.assert_array_equal(base['labels'], y)
    groups = np.array([row['regime'] for row in rows]); target = diagnose(y)
    methods = {'mechanism_world': ['response','latent','response_forecast'],
               'mechanism_telemetry': ['response','history']}
    predictions = {name: np.empty(128,int) for name in [*methods,'latent','response']}
    selections = {name: [] for name in predictions}; splits = []
    for group in sorted(set(groups)):
        held = groups == group; train = ~held
        assert sum(held) == 32
        tx = {name:x[train] for name,x in xs.items()}
        hx = {name:x[held] for name,x in xs.items()}
        inner = [dict(train=np.where(train & (groups != g))[0].tolist(),
                      held=np.where(train & (groups == g))[0].tolist())
                 for g in sorted(set(groups[train]))]
        splits.append(dict(group=int(group), train=np.where(train)[0].tolist(),
                           held=np.where(held)[0].tolist(), inner=inner))
        for name, allowed in methods.items():
            selected, audit = select(tx, y[train], groups[train], allowed)
            predictions[name][held] = predict(tx, y[train], hx, selected)
            selections[name].append(dict(group=int(group), selected=selected, **audit))
        for name in ('latent','response'):
            config, counts = choose(tx[name],y[train],groups[train])
            predictions[name][held] = diagnose(score(fit(tx[name],y[train],config),hx[name]))
            selections[name].append(dict(group=int(group),config=config,inner_correct=counts))
    metrics = {name:stats(pred,rows) for name,pred in predictions.items()}
    for name,pred in predictions.items():
        metrics[name]['bit_errors'] = np.sum(((pred[:,None]>>np.arange(3))&1) != (y>0),axis=0).tolist()
        metrics[name]['combined_correct'] = int(sum((pred==target)&np.array([r['fault']>=4 for r in rows])))
    a=metrics['mechanism_world']
    gate=bool(a['correct']>=116 and a['families']['0']['correct']==16
        and all(v['correct']>=12 for k,v in a['families'].items() if k!='0')
        and a['correct']>=max(metrics[n]['correct'] for n in ('mechanism_telemetry','latent'))+4)
    result=dict(methods=metrics,development_gate=gate,model_retrained=False,
        scope='Readout development only. Model 81 trained on these rows; no independent world-model generalization claim.')
    for name,data in [('summary',result),('predictions',dict(target=target.tolist(),methods={k:v.tolist() for k,v in predictions.items()})),
                      ('selection',selections),('splits',splits),('input-hashes',hashes)]:
        (OUT/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    for p in paths: assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(R))]
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
