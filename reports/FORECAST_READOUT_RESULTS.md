# Do forecast-derived features improve diagnosis?

2026-11-18. [Protocol](FORECAST_READOUT_PROTOCOL.md) · [Evidence](evidence/forecast-readout-development-v1/summary.json) · [Verification](evidence/forecast-readout-development-v1/verification.json).

## Outcome

The preregistered forecast-informed readout did not improve overall diagnosis over the load-response baseline. The development gate failed. No new incidents, training, checkpoint updates, deployments or namespace operations were performed.

| Readout representation | Correct out of 128 |
|---|---:|
| Load response | 104 |
| Load response + forecast changes/residuals (primary) | 104 |
| Full history | 102 |
| Load response + full history | 104 |
| Dreamer latent state | 119 |
| Load response + Dreamer latent state | 119 |

These are **training-data readout-development scores**, not fresh validation results. The frozen world model saw all 128 incidents and their mechanism labels during its training. Its 119/128 latent score does not establish generalization, even though classifier selection and fitting correctly hold out workload groups. The previous fresh-validation result remains Dreamer 56/64 versus load response 60/64. Different datasets and selection rules prevent a controlled comparison of these percentages.

## Comparison and observed tradeoff

All readouts receive four observation windows. The primary adds 12 predicted latency-change features and 12 observed prediction-residual features to 14 load-response features. Residuals use observations only after they arrive. It does not expose future observations to an earlier decision, hidden counters or fault controls.

Nested selection holds out each of four workload regimes for the outer prediction. Each outer training set uses its remaining three regimes for inner selection among the same eight classifier candidates. Every normalizer is fitted only on the corresponding training rows. Independent verification reproduced all configurations, predictions and scores.

Forecast augmentation gets eight incidents right that load response misses, but loses eight that load response gets right:

| Fault family | Load response | With forecast features |
|---|---:|---:|
| Pool only | 10/16 | 9/16 |
| CPU + delay | 6/16 | 11/16 |
| CPU + pool | 9/16 | 10/16 |
| All three faults | 16/16 | 11/16 |

The primary misses the overall requirement of 116/128, the per-family minimum of 12/16, and the requirement to beat load response by four. Adding useful predictions is therefore insufficient, with this representation and classifier family, to improve diagnosis.

## Interpretation

The prior audit still shows that corrected forecasts beat persistence by 67% on the existing validation trajectories. Forecast quality and diagnostic usefulness are different properties. This experiment tests one fixed way of exposing predictions to a diagnostic readout; it does not prove forecasts can never help.

Do not switch the primary outcome to the latent control after seeing these results or launch a long training run merely because this readout failed. The next design decision should address CPU/delay/pool confusion and specify a fresh evaluation. The high latent development score also motivates an evaluation in which the world model itself has not seen held-out incidents, rather than relying on classifier-only cross-validation.

## Reproduction

Development: scripts/develop_forecast_readout.py (requires a fresh output directory).
Independent verification: scripts/verify_forecast_readout.py.
Run with PYTHONPATH=. and OPENBLAS_NUM_THREADS=1.

All inputs are hashed and original evidence is unchanged. No final readout was frozen because the development gate failed.
