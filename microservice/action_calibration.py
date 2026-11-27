"""Grouped calibration of a telemetry forecast, with an optional frozen state term."""
import numpy as np

from microservice.action_outcomes import ACTIONS, PENALTIES, fit_ridge, ridge_predict
from microservice.action_readout import fit_predict
from microservice.action_residual import STRENGTHS, crossfit_errors


def features(actions, baseline, state):
    columns = [actions, actions * baseline[:, None]]
    if state is not None:
        columns.append(state)
    return np.column_stack(columns)


def calibrate(telemetry, state, y, groups, load, held_baseline):
    n = len(y)
    actions = telemetry[:, :len(ACTIONS)]
    baseline = np.empty_like(y)
    modes = ('recalibrated', 'recalibrated_state')
    shifts = {mode: {p: np.empty_like(y) for p in PENALTIES} for mode in modes}
    for group in np.unique(groups):
        test = groups == group
        fit = ~test
        residual = crossfit_errors(telemetry[:n][fit], y[fit], groups[fit])
        baseline[test], _ = fit_predict(telemetry[:n][fit], y[fit], groups[fit], telemetry[:n][test])
        for mode in modes:
            use_state = mode == 'recalibrated_state'
            x = features(actions[:n][fit], y[fit] - residual, state[:n][fit] if use_state else None)
            h = features(actions[:n][test], baseline[test], state[:n][test] if use_state else None)
            for p in PENALTIES:
                shifts[mode][p][test] = ridge_predict(fit_ridge(x, residual, p), h)

    base_error = abs(y - baseline)
    estimates, settings = {}, {}
    for mode in modes:
        options = [(base_error[load].mean(), base_error.mean(), 0., PENALTIES[0])]
        for p in PENALTIES:
            for strength in STRENGTHS[1:]:
                errors = abs(y - np.clip(baseline + strength * shifts[mode][p], 0, 1))
                if errors.mean() <= base_error.mean():
                    options.append((errors[load].mean(), errors.mean(), strength, p))
        _, _, strength, p = min(options)
        settings[mode] = dict(strength=strength, penalty=p if strength else None)
        estimates[mode] = held_baseline.copy()
        if strength:
            use_state = mode == 'recalibrated_state'
            x = features(actions[:n], baseline, state[:n] if use_state else None)
            h = features(actions[n:], held_baseline, state[n:] if use_state else None)
            shift = ridge_predict(fit_ridge(x, y - baseline, p), h)
            estimates[mode] = np.clip(held_baseline + strength * shift, 0, 1)
    return estimates, settings
