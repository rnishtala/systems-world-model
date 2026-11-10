# Independent learned-state classifier confirmation

**Primary result: 80/80; gate passed.** Training seed 23, fresh scenario seed 6801. Training took 12.39 minutes for 100,000 steps. The previous seed-22 80/80 result remains development evidence and is not pooled with this confirmation.

This tests a frozen classifier recipe on an independently trained Dreamer representation. Coefficients are fitted on the original calibration only because a new model has different latent coordinates. No weights, thresholds or normalization were fitted on confirmation incidents.

| Readout | Exact diagnoses | Per-family gate |
|---|---:|---|
| dreamer_auxiliary | 80/80 | pass |
| raw | 78/80 | fail |
| latent (primary) | 80/80 | pass |
| shuffled_latent | 12/80 | fail |
| dreamer_reward | 70/80 | fail |

| Family | Latent primary | Raw telemetry | Reward |
|---|---:|---:|---:|
| Healthy | 10/10 | 10/10 | 10/10 |
| CPU | 10/10 | 10/10 | 10/10 |
| Pool | 10/10 | 10/10 | 9/10 |
| Delay | 10/10 | 10/10 | 10/10 |
| CPU + delay | 10/10 | 10/10 | 10/10 |
| CPU + pool | 10/10 | 8/10 | 1/10 |
| Pool + delay | 10/10 | 10/10 | 10/10 |
| Triple | 10/10 | 10/10 | 10/10 |

## Interpretation

The prespecified independent confirmation passed. This supports a small known-regime triage POC using a classifier over a Dreamer-trained state.
The candidate uses Dreamer to learn the recurrent system representation and a supervised ridge classifier to select the fault set. It does not use imagined reward to make the final diagnosis. Fixed inspections mean this experiment does not demonstrate learned investigation planning or autonomous remediation.
The raw-telemetry baseline and shuffled-label control remain necessary comparisons. One small confirmation batch cannot prove world-model superiority, production reliability or transfer to unseen services. All eight fault regimes were represented in training. Historical calibration includes warm-up measurements, while fresh evaluation uses clean steady-state windows. Startup handling remains untested.

## Next step

Add the frozen classifier to the inference-only API and verify action parity with this evaluation before any namespace-scoped serving promotion. Then evaluate fresh live incidents through that API. LLM orchestration and recovery actions remain separate work.

## Validation and isolation

Three existing classifier tests and four SSH-monitor regression tests passed. Verification reproduced probe coefficients from saved calibration features and every probe prediction from saved test features, checked all runtime label-isolation audits, confirmed checkpoint immutability and validated all 80 measurement windows.
Original service source bundles and all five baseline deployment replicas were restored. No candidate was promoted and no other namespace was modified. The generic evaluator summary describes an individual batch; confirmation status is defined by this preregistered protocol and completion record.

Evidence: `reports/evidence/latent-confirmation/`. Protocol: `reports/LATENT_CONFIRMATION_PROTOCOL.md`.
