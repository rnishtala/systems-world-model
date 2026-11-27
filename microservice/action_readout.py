"""Compare ridge readouts of frozen prior states on the two original masked folds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from microservice.action_forecast import (Forecaster, checkpoint_hash, prefix_frames,
                                         prior_outcomes, RNG_SEED)
from microservice.action_model import validate
from microservice.action_outcomes import ACTIONS, PENALTIES, design, fit_ridge, ridge_predict


def fit_readout(x, y, groups):
    scores = []
    for penalty in PENALTIES:
        predictions = np.empty_like(y)
        for group in np.unique(groups):
            inner = groups != group
            model = fit_ridge(x[inner], y[inner], penalty)
            predictions[~inner] = np.clip(ridge_predict(model, x[~inner]), 0, 1)
        scores.append(np.mean(abs(predictions - y)))
    penalty = PENALTIES[int(np.argmin(scores))]
    return fit_ridge(x, y, penalty), penalty


def fit_predict(x, y, groups, held_x):
    model, penalty = fit_readout(x, y, groups)
    return np.clip(ridge_predict(model, held_x), 0, 1), penalty


def extract(model, rows):
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    @nj.pure
    def infer(frames, action):
        return prior_outcomes(model.agent.model, frames, action, jnp, include_state=True)

    compiled = jax.jit(lambda params, frames, action, key:
                       infer(params, frames, action, seed=key, create=False, modify=False)[1])
    states, native = [], []
    with jax.transfer_guard('allow'):
        for row in rows:
            frames = prefix_frames(row['observations'])
            action = np.eye(len(ACTIONS), dtype=np.float32)[ACTIONS.index(row['action'])]
            values, finite, state = compiled(model.params, frames, 2 * action - 1,
                                             jax.random.PRNGKey(RNG_SEED))
            state = np.asarray(state)
            if not bool(finite) or not np.isfinite(state).all():
                raise ValueError('Nonfinite frozen prior')
            native.append(float(np.clip(np.asarray(values)[0], 0, 1)))
            states.append(np.outer(action, state).ravel())
    return np.asarray(states), np.asarray(native)


def run_fold(source, out, fold):
    arm = source / str(fold) / 'masked'
    train = validate([json.loads(line) for line in (arm / 'train.jsonl').read_text().splitlines()])
    held = validate(json.loads((arm.parent / 'development.json').read_text()))
    train_groups = np.array([r['group'] for r in train])
    held_groups = {r['group'] for r in held}
    expected = json.loads((source / 'protocol.json').read_text())['held_profiles'][fold]
    if held_groups != set(expected) or held_groups.intersection(train_groups):
        raise ValueError('Unexpected outer fold')
    for rows in (train, held):
        if len(rows) != 32 or len({(r['action'], r['group']) for r in rows}) != 32:
            raise ValueError('Expected eight complete four-action profiles')
    model = Forecaster(arm)
    rows = train + held
    state, native = extract(model, rows)
    n = len(train)
    telemetry = np.stack([design(r) for r in rows])
    actions = telemetry[:, :len(ACTIONS)]
    y = np.array([r['outcome'][0] for r in train])
    features = {'telemetry': telemetry, 'state': np.column_stack((actions, state)),
                'combined': np.column_stack((telemetry, state))}
    estimates = {'native': native[n:]}
    penalties = {}
    for name, x in features.items():
        estimates[name], penalties[name] = fit_predict(x[:n], y, train_groups, x[n:])
    from microservice.action_residual import corrected_forecast
    load_fit = np.array([r['action'] == 'load' for r in train])
    state_features = features['state']
    estimates['corrected'], selection = corrected_forecast(
        telemetry[:n], state_features[:n], y, train_groups, load_fit,
        state_features[n:], estimates['telemetry'])
    truth = np.array([r['outcome'][0] for r in held])
    load = np.array([r['action'] == 'load' for r in held])
    metrics = {}
    for name, prediction in estimates.items():
        errors = abs(prediction - truth)
        metrics[name] = {'load': float(errors[load].mean()), 'all': float(errors.mean())}
    reference = json.loads((arm / 'development-scores.json').read_text())
    ref_mae = np.mean([r['prior']['error_fraction']['mae'] for r in reference.values()])
    if not np.isclose(metrics['native']['all'], ref_mae, atol=1e-6) or checkpoint_hash(arm) != model.checkpoint_sha256:
        raise ValueError('Native forecast parity or checkpoint integrity failed')
    adjustments = estimates['corrected'] - estimates['telemetry']
    by_outcome = {}
    for label, mask in [('all_failed', truth == 1), ('all_succeeded', truth == 0)]:
        subset = load & mask
        if subset.any():
            by_outcome[label] = dict(n=int(subset.sum()), mean_adjustment=float(adjustments[subset].mean()))
    result = dict(correction_selection=selection, load_adjustments=by_outcome, metrics=metrics, penalties=penalties, profiles=sorted(held_groups),
                  checkpoint_sha256=model.checkpoint_sha256)
    (out / f'fold-{fold}.json').write_text(json.dumps(result, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1))
    args = parser.parse_args()
    if os.environ.get('POD_NAMESPACE') != 'systems-world-model-lab':
        raise ValueError('Wrong namespace')
    if args.fold is not None:
        run_fold(args.source, args.out, args.fold)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    folds = []
    for i in (0, 1):
        subprocess.run([sys.executable, '-m', 'microservice.action_readout', '--source',
                        str(args.source), '--out', str(args.out), '--fold', str(i)], check=True)
        folds.append(json.loads((args.out / f'fold-{i}.json').read_text()))
    if set(folds[0]['profiles']) & set(folds[1]['profiles']):
        raise ValueError('Duplicated outer development profiles')
    pooled = {name: {key: float(np.mean([f['metrics'][name][key] for f in folds]))
                     for key in ('load', 'all')} for name in folds[0]['metrics']}
    passes = {name: all(f['metrics'][name]['load'] < f['metrics']['telemetry']['load'] for f in folds)
              and pooled[name]['all'] <= pooled['telemetry']['all'] for name in ('state', 'combined', 'corrected')}
    summary = dict(folds=folds, pooled=pooled, passes=passes)
    (args.out / 'summary.json').write_text(json.dumps({'action_readout': summary, 'incidents': []}, indent=2))


if __name__ == '__main__':
    main()
