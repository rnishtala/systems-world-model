"""Cross-fitted telemetry residuals and grouped selection of a state correction."""
import numpy as np
from microservice.action_outcomes import PENALTIES, fit_ridge, ridge_predict

STRENGTHS = (0., .25, .5, 1.)


def crossfit_errors(telemetry, y, groups):
    from microservice.action_readout import fit_predict

    residual = np.empty_like(y)
    for profile in np.unique(groups):
        hold = groups == profile
        prediction, _ = fit_predict(telemetry[~hold], y[~hold], groups[~hold], telemetry[hold])
        residual[hold] = y[hold] - prediction
    return residual


def corrected_forecast(telemetry, state, y, groups, load, held_state, held_baseline):
    from microservice.action_readout import fit_predict

    baseline = np.empty_like(y)
    corrections = {penalty: np.empty_like(y) for penalty in PENALTIES}
    for profile in np.unique(groups):
        hold = groups == profile
        fit = ~hold
        baseline[hold], _ = fit_predict(telemetry[fit], y[fit], groups[fit], telemetry[hold])
        # The validation profile stays outside every fit that constructs these labels.
        residual = crossfit_errors(telemetry[fit], y[fit], groups[fit])
        for penalty in PENALTIES:
            model = fit_ridge(state[fit], residual, penalty)
            corrections[penalty][hold] = ridge_predict(model, state[hold])

    def errors(pred):
        err = np.abs(y - pred)
        return float(err[load].mean()), float(err.mean())

    load_mae, all_mae = errors(baseline)
    candidates = [(load_mae, all_mae, 0., PENALTIES[0])]
    for penalty, correction in corrections.items():
        for strength in STRENGTHS[1:]:
            pred = np.clip(baseline + strength * correction, 0, 1)
            primary, pooled = errors(pred)
            if pooled <= all_mae:
                candidates.append((primary, pooled, strength, penalty))
    primary, pooled, strength, penalty = min(candidates)
    prediction = held_baseline.copy()
    if strength:
        residual = crossfit_errors(telemetry, y, groups)
        model = fit_ridge(state, residual, penalty)
        prediction = np.clip(prediction + strength * ridge_predict(model, held_state), 0, 1)
    return prediction, dict(strength=strength, penalty=penalty if strength else None,
                            inner_load_mae=primary, inner_all_mae=pooled)
