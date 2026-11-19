# Actual six-versus-eight observation experiment

Frozen model91 and byte-identical readouts from the preceding confirmation.64 matched scenarios, each run at both budgets:128 actual incidents. Half retain original conditions; half raise loads3/9 to4/10 and increase CPU/delay severity. Arm order balanced; the same fault seed is replayed within each pair.

| Method | Exact diagnoses |
|---|---:|
| latent6 | 60/64 |
| history6 | 51/64 |
| summary_history6 | 42/64 |
| latent8 | 61/64 |
| history8 | 56/64 |
| summary_history8 | 40/64 |

| Six-window Dreamer regime | Exact diagnoses |
|---|---:|
| Original | 32/32 |
| Higher load/severity | 28/32 |

Preregistered combined feasibility gate: **failed**.
Reliability: False; each-regime gate: False; within two of full controls: True; efficiency: True.

| Mean cost | Six | Eight | Reduction |
|---|---:|---:|---:|
| measured_requests | 39.000 | 52.000 | 25.0% |
| total_requests | 75.000 | 88.000 | 14.8% |
| acquisition_seconds | 4.414 | 5.739 | 23.1% |
| warmup_inclusive_seconds | 8.329 | 9.755 | 14.6% |
| inference_seconds | 0.018 | 0.023 | 20.5% |
| model_inclusive_seconds | 8.348 | 9.778 | 14.6% |

| Compared with | Six-window Dreamer only correct | Control only correct | Paired p |
|---|---:|---:|---:|
| history6 | 11 | 2 | 0.0225 |
| summary_history6 | 18 | 0 | 0.0000 |
| latent8 | 0 | 1 | 1.0000 |
| history8 | 5 | 1 | 0.2188 |
| summary_history8 | 20 | 0 | 0.0000 |

## Scope and limitations

Six-window arms actually stop acquisition at six. No unseen measurements are fabricated. Time measurements include lab request/metrics overhead; warmup-inclusive time starts before fault configuration and excludes the final cleanup reset. Model-inclusive time adds the frozen Dreamer extraction and all three readout computations; graph compilation is primed before collection. These are experimental pipeline timings, not production latency guarantees.

Paired arms share fault controls but execute separately, so their noise and machine state differ. The same topology and fault families remain; only load/severity shifts. Summary-history load bins were frozen for3/9 and remain empty at4/10, although raw history retains actual values. Plain history is the important control under this shift.

No weights were trained, no serving model promoted, and no other namespace modified. The engineering gate does not establish statistical noninferiority or broad reliability. Report all failures; do not tune against this test set.

Protocol: reports/PROBE_LIVE_BUDGET_PROTOCOL.md. Evidence: reports/evidence/probe-live-budget-v1.

## Interpretation and next experiment

Measured acquisition fell from5.74s to4.41s (23.1%); including warmup/configuration,9.75s to8.33s (14.6%). Measured requests fell52 to39 on average (25%); including warmup,88 to75 (14.8%). These savings are observed, not shadow estimates.

Six-window Dreamer scored60/64 (93.75%), versus61/64 at eight and51/64 for plain history at six. Its familiar-regime score remained32/32, but shifted performance fell to28/32. The gate failed because the CPU-plus-delay family scored only1/4 under the shift (5/8 overall). All four six-window errors falsely added pool contention; three were CPU-plus-delay, one delay-only. Eight observations recovered only one of these cases.

The plain-history comparison has unadjusted paired p=0.0225, but several controls were compared, and the load-bin summary control is structurally disadvantaged under the shift. Treat this as exploratory evidence of a representation benefit, not broad superiority.

Next: collect a separate factorial training set that varies load, CPU strength, delay strength, DB hold, and pool size independently with overlapping latency ranges. Include both true and absent pool contention at high delay. Preserve this completed test as held-out audit evidence; do not train on these incidents. Retrain and calibrate on new training/development seeds, freeze again, then evaluate on another untouched shifted set. Keep six observations as the candidate budget; do not promote the current model.

The Kubernetes job took19m44s. Two transient SSH monitoring interruptions did not restart or repeat collection. All128 arms completed within the35s safety bound (maximum15.08s); final health verification confirms serving restoration and no active experiment jobs.
