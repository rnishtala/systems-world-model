# Frozen latent classifier through the live inference API

Saved-trajectory API parity: **80/80**. Label/correctness-reward rejection checks: **passed**.

The serving package contains only model configuration, checkpoint, metadata and frozen classifier coefficients. It excludes calibration, held-out examples and replay. Artifact hashes bind the classifier to its confirmed checkpoint. The server retains its agent role, read-only mounts and no service-account credentials.

Fresh live result: **80/80; gate passed**. Scenario seed 6901; same model 23 and classifier as independent confirmation. No training or fitting occurred.

| Family | Exact diagnoses |
|---|---:|
| Healthy | 10/10 |
| CPU | 10/10 |
| Pool | 10/10 |
| Delay | 10/10 |
| CPU + delay | 10/10 |
| CPU + pool | 10/10 |
| Pool + delay | 10/10 |
| Triple | 10/10 |

Injection-through-diagnosis duration: median 5.81s; p95 8.96s; maximum 9.15s. This includes fault setup, warm-up, four measured batches and four HTTP inference calls.

## Decision and remaining scope

After the experiment restored the baseline, a separate reviewed promotion made the exact tested candidate the lab default. A healthy real-service API smoke check passed; all five deployments are ready. The original inference template and service sources remain available for rollback. Diagnosis-only behavior is retained.
This demonstrates fixed-inspection triage for eight known regimes in one small service chain. It does not establish production reliability, unseen-service generalization, learned investigation planning, LLM orchestration or autonomous remediation. The independent confirmation and this fresh live batch are reported separately.

## Validation and restoration

Seven readout/package tests passed. Evidence verification confirms exact equality of packaged and confirmed classifier arrays, and checks all replay actions, classifier identity, actual API score-to-action mapping, exact offline/live observation equivalence, all fresh window counts and per-family scores. The experiment verified restoration of the original inference template and all five baseline deployments before any separate promotion; no other namespace was modified.

Evidence: `reports/evidence/latent-api/`. Protocol: `reports/LATENT_API_PROTOCOL.md`.
