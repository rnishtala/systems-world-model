# Uncapped trace diagnosis with workload-regime validation

2026-11-18. [Protocol](LAB_REGIME_PROTOCOL.md) · [Evidence](evidence/lab-regime-v1/evaluation/summary.json).

## Design

A new DreamerV3 seed 71 trained on 128 fresh incidents across four load regimes. Latency inputs use log1p(seconds / 0.1), with no hard cap. CPU, pool-wait and database duration counters remain hidden from all methods. Architecture and training budget are unchanged.

Training completed 100,000 steps in 17.0 minutes.
Training used alternating loads 2/4, 4/6, 6/8 and 8/10. Validation used 64 new incidents at 3/7 and 5/9, absent from world-model and readout fitting. Each readout candidate was fitted on training data only; validation selected an existing fit without refitting. The final test, if warranted, uses new mixed sequences 3/11/6/9 and reverse.

## Validation used for readout selection

| Method | Exact diagnoses |
|---|---:|
| Full-history telemetry | 49/64 |
| Latest snapshot | 47/64 |
| Dreamer state + readout | 50/64 |
| Shuffled-label control | 10/64 |
| Dreamer auxiliary head | 43/64 |

Preregistered screening gate: **failed**. Requires Dreamer >=58/64, all eight healthy correct, >=6/8 in every other family, and no more than two fewer correct than the stronger telemetry baseline.

These figures selected the readouts, so they are not an independent final-test estimate. The world model itself did not train on validation incidents. Validation paired statistics in the evidence are descriptive after selection, not confirmatory significance tests.

| Fault family | Dreamer state | History | Latest |
|---|---:|---:|---:|
| Healthy | 8/8 | 8/8 | 8/8 |
| CPU | 8/8 | 7/8 | 8/8 |
| Pool | 5/8 | 7/8 | 2/8 |
| Delay | 4/8 | 3/8 | 5/8 |
| CPU + delay | 4/8 | 0/8 | 3/8 |
| CPU + pool | 8/8 | 8/8 | 6/8 |
| Pool + delay | 7/8 | 8/8 | 7/8 |
| All three | 6/8 | 8/8 | 8/8 |

## Final test deferred

The screening gate failed. No final-test incidents were collected, saving 80 live fault injections. No final-test accuracy or generalization advantage is claimed.

## Interpretation and next experiment

Dreamer diagnosed 50/64 validation incidents (78.1%), only one more than full-history telemetry (49/64, 76.6%). Healthy and CPU-only cases were all correct, but pool-only (5/8), delay-only (4/8), and CPU-plus-delay (4/8) missed the per-family threshold. This does not establish added value from the world model.

The previous experiment used different incidents, workloads and selection rules. Its 54/80 score cannot serve as a controlled before/after comparison. Uncapped scaling and broader coverage were changed together, so their individual effects are unknown.

Before another training run, use the existing training data and these now-development validation cases to examine confusion and overlapping traces across load regimes. Check whether pool contention and injected delay are distinguishable from the available observations. Then preregister one targeted change and collect fresh validation data. Do not increase architecture size or reuse these scores as independent confirmation.

All requests in this experiment completed successfully: it evaluates performance-fault triage, not outage or failed-request diagnosis.

## Verification and limits

Independent verification reproduced training-only readouts, validation selection and saved predictions; checked measured observations, workload balancing, split timing, hashes and wait-action alignment. The upstream checkpoint passed label-invariance checks. The prior inference template and replica count were restored; temporary resources were removed. No model was promoted. Latent features were not regenerated locally.

This is one model seed, eight known controlled fault families, and fixed observation budgets. It does not establish autonomous investigation, unseen-fault diagnosis, remediation, or a unique benefit from generative world-model training over other recurrent representations.
