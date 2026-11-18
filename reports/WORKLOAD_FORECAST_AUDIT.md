# Forecast audit: decoder units corrected

2026-11-18. [Evidence](evidence/workload-forecast-audit-v1/summary.json) · [Verification](evidence/workload-forecast-audit-v1/verification.json).

## Result

The earlier conclusion that Dreamer forecasts were worse than persistence was caused by a forecast adapter bug. The same frozen checkpoint predicts the next trace latencies better than persistence after unit conversion. No retraining or new incident collection was needed.

| Split | Original, wrong-unit MSE | Corrected forecast MSE | Persistence MSE |
|---|---:|---:|---:|
| Training | 0.232797 | 0.018230 | 0.040769 |
| Validation | 0.251435 | 0.032954 | 0.099973 |

Corrected validation MSE is **67.0% lower** than persistence. MSE is measured in the observation's log-latency units (`log1p(seconds / 0.1)`), not seconds. These are the existing validation incidents, not a new independent test.

## Cause and fix

The pinned upstream `Decoder` uses a `symlog_mse` vector head. Its loss compares the decoder output with `symlog(target)`, but `MSE.pred()` returns the compressed output directly. Our adapter incorrectly compared that output to observation values.

The adapter now applies `sign(x) * expm1(abs(x))` when the decoder's `symlog` flag is enabled. It converts each particle before averaging. This removes only Dreamer's internal compression; it does not undo the experiment's external log-latency observation transform. Uncompressed decoders remain unchanged.

The bug affects the shared `Model.forecast` adapter; older reports using that adapter may also contain wrong-unit forecasts and require separate review. This audit verifies model 81 only. The deployed serving process was not updated; the repository fix was exercised in an isolated audit process and the existing inference deployment was restored unchanged.

## Alignment and reconstruction checks

The audit replayed all 128 training and 64 validation incidents from the same checkpoint, reproducing every original raw forecast within numerical tolerance. The prior consumes the workload action for the next observation, with the actual normalized action checked in the carry. The fixed production adapter agrees with separately decoded prior predictions.

Corrected posterior reconstruction MSE is 0.021801 on training and 0.029404 on validation. Corrected prior forecast MSE is 0.018230 and 0.032954 respectively. Corrected forecasts outperform persistence in every latency channel and at each of the three scored transition positions on validation. The audit found no additional transition-offset error in this path; it does not establish correctness of every other forecast consumer.

Checkpoint hashes before and after match the original experiment. Audit source was copied into a temporary filesystem; the results PVC was read-only. No service requests, fault controls, optimizer steps or checkpoint writes occurred. Export hashes and all reported metrics were independently verified locally. The inference template and replica count were restored, and the audit pod was deleted.

## Triage conclusion and next step

Diagnosis is unchanged: **Dreamer 56/64; load-response baseline 60/64**. The forecast criterion now passes, but the overall screening gate still fails because diagnosis misses 58/64 and trails the strongest baseline by four incidents. Skipping the final test remains the correct decision. Original artifacts remain intact; this report and separate audit evidence correct their interpretation.

The next focused experiment should test whether forecast-derived features, such as predicted latency changes or prediction residuals, improve diagnosis beyond the existing latent readout. First develop that readout using training data, with equal observations for all baselines, then freeze it before fresh validation. Do not claim autonomous investigation, counterfactual planning, or a unique advantage over other learned predictors from this audit alone.

## Reproduction

- `scripts/audit_workload_forecast.py`: frozen-checkpoint replay and adapter equivalence checks.
- `scripts/run_workload_forecast_audit.py`: namespace-bounded, read-only-PVC execution and restoration.
- `scripts/verify_workload_forecast_audit.py`: local metric, hash and gate verification.
- `tests/test_decoder_units.py`: inverse-transform and particle-aggregation checks.
