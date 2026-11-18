# Workload-conditioned Dreamer results

2026-11-18. [Frozen protocol](LAB_WORKLOAD_PROTOCOL.md). [Evidence](evidence/lab-workload-v2/evaluation/summary.json).

Training completed 100,000 steps in 18.0 minutes using the original 128 training incidents. Seed 81 was trained from scratch. Fresh validation used 64 incidents with seed 8802.

The state transition now receives a continuous planned-workload action. The declared next workload is also an observation feature; this experiment does not isolate the action pathway from that observation pathway. Forecasts are made before ingesting the observation being predicted. No future telemetry or fault labels enter posterior inference.

## Validation used for selection

| Method | Exact diagnoses |
|---|---:|
| Full trace history | 48/64 |
| Latest snapshot | 47/64 |
| Load-response features | 60/64 |
| Dreamer state | 56/64 |
| Shuffled-label state control | 4/64 |
| Auxiliary head | 55/64 |

Combined screening gate: **failed**. Forecast gate: **failed**.

These validation scores select readouts and are not independent final-test estimates. All candidate fits and normalizers use training data only; selected fits are not refitted.

| Fault family | Dreamer | History | Load response |
|---|---:|---:|---:|
| Healthy | 8/8 | 8/8 | 8/8 |
| CPU | 8/8 | 3/8 | 8/8 |
| Pool | 6/8 | 8/8 | 8/8 |
| Delay | 6/8 | 8/8 | 8/8 |
| CPU + delay | 8/8 | 4/8 | 6/8 |
| CPU + pool | 6/8 | 8/8 | 7/8 |
| Pool + delay | 8/8 | 3/8 | 8/8 |
| All three | 6/8 | 6/8 | 7/8 |

## Forecasting

MSE across four log-latency channels and three real transitions per incident. Lower is better; persistence predicts the current latency at the next workload. No terminal bookkeeping transitions are scored.

| Split | Dreamer forecast MSE | Persistence MSE |
|---|---:|---:|
| train | 0.232797 | 0.040769 |
| validation | 0.251435 | 0.099973 |

## Final test skipped

The preregistered screening gate failed. No 80-incident final test was collected.

## Interpretation and next step

Dreamer scored 56/64 (87.5%) on fresh validation used for readout selection, behind the load-response baseline at 60/64 (93.8%). It satisfies the family minimum of six out of eight and healthy eight out of eight, but misses the overall 58/64 requirement and trails the strongest baseline by four rather than at most two.

Forecast MSE is approximately 2.52 times persistence on validation and 5.71 times persistence on the training trajectories. This fails the prediction criterion even on familiar trajectories. A diagnostic representation is being learned, but useful workload prediction has not been demonstrated.

Before another long run, audit the forecasting path: compare posterior reconstruction with one-step prior forecasts on saved training trajectories, verify decoder units and time alignment, and inspect channel-wise error. If that path is correct, investigate the predictive objective and replay coverage. Do not assume a larger model or more steps will fix the failure.

The earlier 50/64 result used a different validation batch and different training contract. It is not a controlled before/after comparison.

## Verification and limitations

The first attempt stopped at upstream scalar-action dtype preflight before training or live collection. The corrected run uses a one-element float32 action array. Actual upstream normalization, rejection of unmeasured transitions, a separate 256-step smoke checkpoint, label isolation, and evaluation action carry alignment were checked. Independent local verification reproduced feature construction, readout selection, predictions, forecast targets/errors, split timing and artifact hashes. Latent features and forecasts were not regenerated locally.

Architecture size and full-run step budget stayed fixed, but action contract, replay policy, next-workload observation, rewards and episode termination changed together. This is not a single-variable causal ablation. Training sees only measured trajectories, not arbitrary intervention outcomes. It does not establish counterfactual planning, autonomous remediation, unseen-fault diagnosis or unique value over other recurrent models.

The previous inference template and replica count were restored. No checkpoint was promoted. All work remained in the systems-world-model repository and lab namespace.
