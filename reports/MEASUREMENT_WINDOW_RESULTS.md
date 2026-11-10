# Post-warm-up measurement results

The measurement-only reset excludes completed warm-up requests while preserving fault configuration, lease expiry and pooled database connections. The previous failed run is preserved unchanged.

The trained model and calibration were fixed. The historical calibration still includes warm-up samples: this experiment isolates a measurement change, not a retraining improvement.

## Model seed 22

| Readout | Exact diagnoses | Gate |
|---|---:|---|
| dreamer_auxiliary | 78/80 | pass |
| raw | 78/80 | fail |
| latent | 80/80 | pass |
| shuffled_latent | 8/80 | fail |
| dreamer_reward | 77/80 | fail |

| Family | Dreamer reward | Auxiliary | Telemetry |
|---|---:|---:|---:|
| Healthy | 10/10 | 10/10 | 10/10 |
| CPU | 10/10 | 10/10 | 10/10 |
| Pool | 10/10 | 10/10 | 10/10 |
| Delay | 10/10 | 10/10 | 10/10 |
| CPU + delay | 10/10 | 10/10 | 10/10 |
| CPU + pool | 8/10 | 9/10 | 8/10 |
| Pool + delay | 10/10 | 10/10 | 10/10 |
| Triple | 9/10 | 9/10 | 10/10 |

## Decision and limits

Development gate: failed. Independent confirmation: skipped after development failure.
No candidate was promoted. All original service sources and five deployment replicas were restored. No other namespace was modified.

This is a small experiment across eight known fault regimes, not a production reliability estimate. Startup transients were intentionally excluded and need their own scenario. The telemetry-only comparator is essential: a high Dreamer score alone does not demonstrate an advantage from its learned world model. LLM-assisted resolution and unseen-system generalization remain untested.

All three primary errors omitted pool contention: two CPU + pool incidents and one triple. The frozen latent-state classifier scored 80/80, while the shuffled-label control scored 8/80. This suggests the learned representation contains diagnostic information that the reward readout does not consistently use. It remains a secondary finding from one small batch, not an independent validation or a retroactive primary success. Next: prespecify this fixed latent readout and test it with an independent model and fresh incidents.

Validation: four measurement/control-boundary tests and four SSH-monitor regression tests passed. Runtime label-isolation audits, checkpoint immutability and all 80 incident window counts are preserved with the results.

Evidence: `reports/evidence/measurement-window/`. Protocol: `reports/MEASUREMENT_WINDOW_PROTOCOL.md`.
