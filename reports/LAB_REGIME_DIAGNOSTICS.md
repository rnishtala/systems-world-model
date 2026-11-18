# What the trace-only model misses

2026-11-18. Offline development analysis of the completed lab-regime-v1 data. No new incidents, model updates, deployments, or namespace changes.

## Finding

The available traces carry a useful load-response signal. The current Dreamer state readout is not reliably extracting it. This is evidence against declaring pool-only and delay-only faults indistinguishable; it is not proof that every combination can be identified.

| Representation | Correct on reused validation |
|---|---:|
| Dreamer state | 50/64 |
| Full trace history | 49/64 |
| Latest snapshot | 47/64 |
| Low/high-load trace levels plus response slopes | 59/64 |

The exploratory representation uses only observed request counts and the same four latency channels. It averages the two low-load and two high-load observations, and computes their log-latency difference per additional concurrent request. Its 14 features use no fault labels, controls, CPU counters, pool counters, database duration counters, or regime identifiers. All normalization and classifier fits use the original 128 training incidents. The same eight classifier candidates are selected using the already-used 64 validation incidents.

**The 59/64 score is development evidence after feature design and readout selection. It must not be presented as independent test accuracy or compared statistically as a confirmed improvement.** It still fails reliability: CPU-plus-delay is only 3/8; every other family is 8/8.

## Where the errors occur

Dreamer confuses four of eight delay-only incidents with pool-only faults, and two pool-only incidents with delay. It also calls four CPU-plus-delay incidents all-three faults; in the opposite direction it misses pool contention in two all-three cases.

Ledger log-latency response per additional request is clearly separated for the observed pure faults:

| Family | Training range | Validation range |
|---|---:|---:|
| Pool only | 0.0349–0.2000 | 0.0656–0.1441 |
| Delay only | -0.0033–0.0222 | 0.00002–0.0164 |

For CPU-plus-delay versus all-three faults, those response ranges overlap. Upstream CPU work occurs before downstream requests in the service implementation, so it can change the arrival pattern at the ledger. Whether that explains the overlap is a hypothesis, not a measured causal conclusion. The simple response classifier also remains imperfect despite using this signal.

Thirteen of 64 validation traces are closer to a different training fault family than their own in training-standardized full-history distance. That is a descriptive measure of representation overlap, not a proof of physical non-identifiability.

## Next focused experiment

Keep architecture size and the 100,000-step budget fixed. Make load response an explicit learning target: predict the next trace observation conditioned on the next observed workload, and evaluate whether the resulting state retains load-response information. The current extraction uses wait actions throughout, so this requires explicit transition/action alignment; workload must be provided before the observation it produces. Do not silently treat a future workload as an already-observed input.

Before a live run, implement and verify that alignment on synthetic trajectories. Compare the trained state against both existing telemetry baselines and the stronger load-response baseline above, with equal observations. Use only existing training data for fitting; these validation data are now development data. Freeze the new implementation, candidate selection rules and reliability criteria before collecting fresh validation. Collect a final test only after the per-family screening gate passes. No serving-model promotion.

This defines the next development direction, not a completed preregistration or a launched training run. A larger model, more repetitions of the same run, or exposing direct fault counters is not justified by this analysis.

## Reproduction

Run `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/analyze_lab_regime.py`.

[Machine-readable evidence](evidence/lab-regime-v1/development-analysis.json) includes confusion matrices, feature-selection counts, response ranges and nearest-training-trace distances. The original experiment artifacts and scores remain unchanged.
