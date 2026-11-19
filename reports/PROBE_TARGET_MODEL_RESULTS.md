# Probe-target-aware Dreamer result

2026-11-18. [Protocol](PROBE_TARGET_MODEL_PROTOCOL.md) · [Evidence](evidence/probe-target-model-v1/evaluation/summary.json).

The prespecified feasibility gate passed. A new upstream DreamerV3 checkpoint (seed 91) was trained from scratch with probe target and concurrency as actions. The serving checkpoint was not changed.

Training used 64 measured eight-observation trajectories and 100,000 environment steps, taking 17.2 minutes. These trajectories came from the earlier probe pilot, including its former test set; they are training data in this new experiment. The comparison below uses 32 newly collected incidents (seed 9202), after readouts and checkpoint were frozen.

| Method | Exact diagnoses |
|---|---:|
| Dreamer latent readout (primary) | 32/32 |
| Telemetry response + ledger probe | 32/32 |
| Full observation history | 32/32 |
| Response + history | 32/32 |
| Dreamer auxiliary head | 32/32 |
| Shuffled-label latent control | 4/32 |

| Criterion | Result |
|---|---|
| Primary reliability thresholds | Pass |
| Within one correct diagnosis of strongest telemetry control | Pass |
| Ledger forecast MSE below persistence | Pass |

Reliability thresholds are >=29/32 overall, healthy 4/4 and each fault family >=3/4. They are screening criteria, not confidence bounds on production reliability. Passing feasibility would not establish diagnostic superiority.

| Family | Dreamer primary | Response control |
|---|---:|---:|
| Healthy | 4/4 | 4/4 |
| CPU | 4/4 | 4/4 |
| Pool | 4/4 | 4/4 |
| Delay | 4/4 | 4/4 |
| CPU + delay | 4/4 | 4/4 |
| CPU + pool | 4/4 | 4/4 |
| Pool + delay | 4/4 | 4/4 |
| All three | 4/4 | 4/4 |

## Measured-action forecasting

| Transition subset | Count | Dreamer MSE | Persistence MSE |
|---|---:|---:|---:|
| all | 224 | 0.016288 | 0.105967 |
| same_target | 176 | 0.018190 | 0.100924 |
| switched_target | 48 | 0.009315 | 0.124456 |

Forecasts predict next ledger latency in log1p(seconds/0.1) units. Decoder symlog is inverted per particle before averaging. Comparisons cover measured actions only; no unobserved counterfactual effect or autonomous probe selection was tested. Persistence is a naive forecast control; this does not demonstrate superiority over a learned forecasting baseline.

## Paired diagnosis comparisons

- Against response: Dreamer-only correct 0, control-only correct 0; two-sided exact McNemar p=1.0000.
- Against history: Dreamer-only correct 0, control-only correct 0; two-sided exact McNemar p=1.0000.
- Against response_history: Dreamer-only correct 0, control-only correct 0; two-sided exact McNemar p=1.0000.

## Interpretation

The new model meets the small-sample feasibility criteria for measured probe transitions and diagnosis. The useful next question is whether model predictions reduce probe count or observation cost under a fixed budget, followed by a larger frozen evaluation. Equally informed telemetry controls remain essential; matching their accuracy alone is not evidence that a world model is necessary.

## Verification and limits

The actual upstream smoke run checked normalized two-dimensional actions, rejection of unmeasured replay actions, checkpoint loading, posterior extraction, prior forecasts, label exclusion and the offline-only serving guard. Local verification reproduced training-only readout selection/fits, input observations, action alignment, predictions and metrics; checked source/export hashes, data provenance and checkpoint/readout immutability assertions. Latent states and prior forecasts were not regenerated locally.

Both target choices are encoded as endpoints of a bounded float action component; replay accepts only the measured endpoint and concurrency. Dreamer uses its recurrent world model and auxiliary mechanism supervision. Its actor is overridden by the measured acquisition schedule, and reward is zero: this is not a learned investigation policy. The encoded observation also includes the declared next action. Before testing alternative-action planning, separate that scheduling metadata from the encoded state and validate action effects on randomized measured transitions.

The experiment uses one model seed, 64 training and 32 test incidents, the same eight controlled performance-fault families and 3/9 concurrency values. All requests succeeded. Pilot-derived observations and protocol choices influenced this research direction. Development readout CV is not independent world-model evaluation because the model saw those training trajectories. Small sample size, fixed topology, sequential probes and previously studied fault families limit generalization claims. The older workload-only model used different observations and test incidents, so these scores are not a controlled before/after accuracy improvement.

All compute and fault injection stayed in systems-world-model-lab. Inference was restored to its exact prior template and replica count, experiment resources were removed, and the existing serving model was not promoted or replaced.
