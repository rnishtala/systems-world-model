# Fresh confirmation with frozen readouts

2026-11-18. [Protocol](FROZEN_CONFIRMATION_PROTOCOL.md) · [Evidence](evidence/lab-confirm-v1/evaluation/summary.json).

The unchanged model-81 checkpoint and all readouts were frozen before 64 fresh incidents (seed 9002) were collected. Readout configuration selection and fitting used only the original 128 training incidents. No model training or validation-time tuning occurred.

This is a new confirmation motivated by development findings, not a reinterpretation of the failed forecast-feature development gate. The primary was fixed to the latent-state readout before collection.

| Method | Exact diagnoses | 95% Wilson interval |
|---|---:|---:|
| Load response | 58/64 | 81.0%–95.6% |
| Full history | 46/64 | 59.9%–81.4% |
| Dreamer latent state (primary) | 53/64 | 71.8%–90.1% |
| Load response + history | 57/64 | 79.1%–94.6% |
| Load response + latent state | 53/64 | 71.8%–90.1% |
| Load response + forecast features | 59/64 | 83.0%–96.6% |
| Dreamer auxiliary head | 53/64 | 71.8%–90.1% |

| Criterion | Result |
|---|---|
| Primary diagnosis reliability | Fail |
| At least four more correct than strongest telemetry control | Fail |
| Forecast MSE below persistence | Pass |
| Combined gate | Fail |

Reliability requires >=58/64 overall, healthy 8/8 and every other family >=6/8. These are prespecified sample thresholds, not confidence bounds on production reliability.

| Family | Dreamer primary | Load response | Forecast-feature readout |
|---|---:|---:|---:|
| Healthy | 8/8 | 8/8 | 8/8 |
| CPU | 8/8 | 7/8 | 8/8 |
| Pool | 5/8 | 8/8 | 8/8 |
| Delay | 8/8 | 8/8 | 8/8 |
| CPU + delay | 4/8 | 4/8 | 6/8 |
| CPU + pool | 5/8 | 7/8 | 5/8 |
| Pool + delay | 8/8 | 8/8 | 8/8 |
| All three | 7/8 | 8/8 | 8/8 |

## Forecasts and paired diagnosis comparisons

Corrected forecast MSE: 0.029793; persistence: 0.079490. Relative error reduction: 62.5%. Units are the experiment's log-latency observations, not seconds.
- Against Load response: primary-only correct 3, baseline-only correct 8; two-sided exact McNemar p=0.2266.
- Against Full history: primary-only correct 14, baseline-only correct 7; two-sided exact McNemar p=0.1892.
- Against Load response + history: primary-only correct 4, baseline-only correct 8; two-sided exact McNemar p=0.3877.

These are paired comparisons on one fresh batch. Control comparisons are not corrected for multiplicity; do not turn sample differences into a broad superiority claim.

## Interpretation

The combined criterion failed. This experiment does not establish reliable diagnostic superiority from the world model. Do not change the primary to whichever secondary readout scored best after observing this batch. No further fault collection or training was launched.

The forecast-feature control scored 59/64, only one more correct than load response, and identified CPU-plus-pool faults in only 5/8 cases. It does not rescue the failed primary result or meet the per-family minimum. The primary missed pool-only (5/8), CPU-plus-delay (4/8), and CPU-plus-pool (5/8) cases. A next experiment should target these diagnostic ambiguities rather than repeat broad training without a specific hypothesis.


## Verification and limits

Independent verification reproduced training-only configuration selection and readout fits, checked freeze-before-collection timing, workload balancing and measurement validity, reproduced predictions and metrics, and verified export hashes. The job checked checkpoint and readout immutability, label isolation and workload-action alignment. Latent states and forecasts were not regenerated locally.

The world model and readouts did not train on these incidents. Fault classes remain the same eight controlled families, all measured requests succeeded, and only one checkpoint and batch were assessed. Earlier validation influenced the research direction, but no confirmation data influenced this frozen fit. This does not establish unseen-fault diagnosis, autonomous investigation or remediation.

Work remained inside the systems-world-model repository and namespace. Inference was restored to its original template and replica count; temporary resources were removed.
