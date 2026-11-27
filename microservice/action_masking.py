"""Paired two-fold development experiment for the repair outcome loss."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from microservice.action_model import validate, WIDTH
from microservice.action_outcomes import ACTIONS
from microservice.action_forecast import verify_source
from microservice.action_diagnostic import score_rows, summarize
from microservice.temporal_calibration import FEATURES


def folds(rows):
    groups = {}
    for row in validate(rows):
        groups.setdefault(row['group'], []).append(row['action'])
    regimes = {}
    for group, actions in groups.items():
        regime, suffix = group.rsplit('-', 1)
        if Counter(actions) != Counter(ACTIONS) or suffix not in ('0', '1'):
            raise ValueError('Each profile must have all four actions and a known replicate')
        regimes.setdefault(regime, set()).add(suffix)
    if len(regimes) != 8 or any(pair != {'0', '1'} for pair in regimes.values()):
        raise ValueError('Expected two profiles in each of eight regimes')
    for suffix in ('0', '1'):
        held = {g for g in groups if g.endswith('-' + suffix)}
        yield ([r for r in rows if r['group'] not in held],
               [r for r in rows if r['group'] in held])


def mask_errors(errors, target, xp):
    telemetry = target[..., -2:-1]
    outcome = target[..., -1:]
    boundary = len(FEATURES)
    return xp.concatenate((errors[..., :boundary] * telemetry,
                           errors[..., boundary:-2] * outcome,
                           errors[..., -2:]), axis=-1)


@contextmanager
def masked_objective():
    from embodied.jax.outs import MSE
    import jax.numpy as jnp
    original = MSE.loss
    seen = False

    def loss(output, target):
        nonlocal seen
        errors = original(output, target)
        # Vector is the sole width-57 observation in this agent. Apply before
        # upstream channel reduction, preserving each valid error's coefficient.
        if target.ndim == 3 and target.shape[-1] == WIDTH:
            seen = True
            return mask_errors(errors, target, jnp)
        return errors

    MSE.loss = loss
    try:
        yield
        if not seen:
            raise RuntimeError('No vector reconstruction loss was masked')
    finally:
        MSE.loss = original


def run_arm(root, fold, mode, seed):
    from microservice.action_train import train_partition
    from microservice.configuration import elements  # establishes pinned upstream path

    fit = root / fold / mode
    meta = json.loads((fit / 'manifest.json').read_text())
    with masked_objective() if mode == 'masked' else nullcontext():
        train_partition(fit, meta, seed, 25000)


def score_arm(root, fold, mode):
    fit = root / fold / mode
    verify_source(fit)
    held = json.loads((root / fold / 'development.json').read_text())
    scores = score_rows(fit, held)
    by_profile = {}
    for group in sorted({r['group'] for r in held}):
        indices = [i for i, r in enumerate(held) if r['group'] == group]
        by_profile[group] = summarize([scores[i] for i in indices],
                                      [held[i]['outcome'] for i in indices])
    (fit / 'development-scores.json').write_text(json.dumps(by_profile, allow_nan=False))


def experiment(source, root, seed):
    manifest = verify_source(source)
    rows = [json.loads(line) for line in (source / 'train.jsonl').read_text().splitlines()]
    partitions = list(folds(rows))
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(source=source.parent.name, source_sha256=manifest['training_sha256'],
                steps_per_arm=25000, seed=seed, primary='prior error-fraction MAE',
                normalization='Drop padded errors before the original reduction. Preserve valid coefficients.',
                held_profiles=[sorted({r['group'] for r in held}) for _, held in partitions])
    (root / 'protocol.json').write_text(json.dumps(plan, indent=2))
    profile_scores = {'control': {}, 'masked': {}}
    for i, (train, held) in enumerate(partitions):
        folder = root / str(i)
        folder.mkdir()
        (folder / 'development.json').write_text(json.dumps(held))
        data = ''.join(json.dumps(row) + '\n' for row in train).encode()
        for mode, scores in profile_scores.items():
            fit = folder / mode
            fit.mkdir()
            (fit / 'train.jsonl').write_bytes(data)
            meta = dict(manifest, training_sha256=hashlib.sha256(data).hexdigest(),
                        training_trials=len(train), training_profiles=8, excluded_trials=len(held))
            (fit / 'manifest.json').write_text(json.dumps(meta))
            print(f'Fold {i+1}/2 {mode}: training and scoring', flush=True)
            command = [sys.executable, '-m', 'microservice.action_masking', '--out', str(root),
                       '--fold', str(i), '--arm', mode, '--seed', str(seed)]
            subprocess.run(command, check=True)
            subprocess.run(command + ['--score'], check=True)
            scores.update(json.loads((fit / 'development-scores.json').read_text()))
    metrics = {}
    for mode, scores in profile_scores.items():
        metrics[mode] = {branch: {target: float(np.mean([s[branch][target]['mae'] for s in scores.values()]))
                                  for target in next(iter(scores.values()))[branch]}
                         for branch in ('prior', 'reconstruction')}
    differences = [profile_scores['masked'][g]['prior']['error_fraction']['mae'] -
                   score['prior']['error_fraction']['mae'] for g, score in profile_scores['control'].items()]
    summary = dict(metrics=metrics, profiles=len(differences), trials=len(rows),
                   improved_profiles=sum(d < 0 for d in differences), paired_delta=float(np.mean(differences)))
    (root / 'summary.json').write_text(json.dumps({'action_masking': summary, 'incidents': []}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--fold', choices=('0', '1'))
    parser.add_argument('--arm', choices=('control', 'masked'))
    args = parser.parse_args()
    if os.environ.get('POD_NAMESPACE') != 'systems-world-model-lab':
        raise ValueError('Wrong namespace')
    if args.score:
        score_arm(args.out, args.fold, args.arm)
    elif args.arm:
        run_arm(args.out, args.fold, args.arm, args.seed)
    else:
        experiment(args.source, args.out, args.seed)


if __name__ == '__main__':
    main()
