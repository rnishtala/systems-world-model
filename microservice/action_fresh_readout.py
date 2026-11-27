"""Freeze the paired readouts before collecting a new stratified incident batch."""
import hashlib
import json
from pathlib import Path

import numpy as np
from microservice.action_model import validate
from microservice.action_residual import apply_correction, fit_correction
from microservice.action_outcomes import ACTIONS, design, ridge_predict
from microservice.action_readout import extract, fit_readout


def matrices(model, inputs, residual=False):
    state, _ = extract(model, inputs)
    telemetry = np.stack([design(r) for r in inputs])
    if residual:
        return {'telemetry': telemetry, 'state': np.column_stack((telemetry[:, :len(ACTIONS)], state))}
    return {'telemetry': telemetry, 'combined': np.column_stack((telemetry, state))}


def freeze(model, source, out, residual=False):
    rows = validate([json.loads(line) for line in (source / 'train.jsonl').read_text().splitlines()])
    groups = np.array([r['group'] for r in rows])
    if len(rows) != 64 or len(set(groups)) != 16:
        raise ValueError('Expected full training partition')
    y = np.array([row['outcome'][0] for row in rows])
    models, penalties = {}, {}
    features = matrices(model, rows, residual)
    choice = None
    if residual:
        models['telemetry'], penalties['telemetry'] = fit_readout(features['telemetry'], y, groups)
        load = np.array([r['action'] == 'load' for r in rows])
        models['correction'], choice = fit_correction(features['telemetry'], features['state'], y, groups, load)
        models['strength'] = choice['strength']
    else:
        for name, x in features.items():
            models[name], penalties[name] = fit_readout(x, y, groups)
    raw = json.dumps(models, allow_nan=False).encode()
    (out / 'frozen-readouts.json').write_bytes(raw)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), penalties=penalties, correction=choice,
                primary='Post-load error-rate MAE', guardrail='All-action MAE',
                decision='Lower load MAE and no higher all-action MAE versus telemetry',
                uncertainty='Paired bootstrap across ten workload profiles; 10000 draws, seed 777')


def score(model, rows, out, identity):
    raw = (out / 'frozen-readouts.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity['sha256']:
        raise ValueError('Readout identity changed')
    models = json.loads(raw)
    inputs = [{k:r[k] for k in ('action','observations')} for r in rows]
    residual = identity.get('correction') is not None
    features = matrices(model, inputs, residual)
    telemetry = np.clip(ridge_predict(models['telemetry'], features['telemetry']), 0, 1)
    candidate = 'corrected' if residual else 'combined'
    if residual:
        prediction = apply_correction(models['correction'], models['strength'], features['state'], telemetry)
    else:
        prediction = np.clip(ridge_predict(models['combined'], features['combined']), 0, 1)
    forecasts = {'telemetry': telemetry, candidate: prediction}
    reference = [r['predictions']['snapshot_ridge'][0] for r in rows]
    np.testing.assert_allclose(forecasts['telemetry'], reference, rtol=1e-8, atol=1e-9)
    truth = np.array([r['outcome'][0] for r in rows])
    groups = np.array([r['group'] for r in rows])
    actions = np.array([r['action'] for r in rows])
    error = {name: abs(pred-truth) for name,pred in forecasts.items()}
    report = {}
    for action in ['all', *sorted(set(actions))]:
        keep = np.ones(len(rows),bool) if action == 'all' else actions == action
        diff = error[candidate] - error['telemetry']
        paired = np.array([diff[keep & (groups==g)].mean() for g in np.unique(groups[keep])])
        draws = np.random.default_rng(777).choice(paired,(10000,len(paired)),replace=True).mean(1)
        report[action] = {name:float(e[keep].mean()) for name,e in error.items()}
        report[action].update(delta=float(paired.mean()), wins=int((paired<0).sum()),
                              profiles=len(paired), interval=np.quantile(draws,[.025,.975]).tolist())
    for i,row in enumerate(rows):
        row['readout_predictions'] = {name:float(v[i]) for name,v in forecasts.items()}
    return dict(candidate=candidate, metrics=report, passes=report['load']['delta']<0 and report['all']['delta']<=0,
                telemetry_parity=True, frozen_sha256=identity['sha256'])
