# Fresh lab diagnosis comparison

**Outcome: no prespecified added-value signal.**

Fresh physical incidents from checkout → payment → ledger → PostgreSQL: 80 training and 80 later evaluation cases. Ten cases per healthy/single/pair/triple fault family. Training mixed ordinary and shifted load/severity; evaluation used shifted settings and a different scenario seed. All data stayed in the isolated lab.

Actual upstream DreamerV3 trained from scratch for 100,000 steps (seed 51) with training-only mechanism supervision. Its state readout and the full-history telemetry baseline used the same observation sequence, penalty-1 ridge fitting, and threshold zero. Both readouts were frozen before evaluation collection. CPU, pool and trace checks were fixed, not learned.

| Method | Correct / 80 | Reliability gate |
|---|---:|---|
| Dreamer state + ridge (primary) | 80 | Pass |
| Full observation history + ridge (primary baseline) | 79 | Pass |
| Last observed telemetry + ridge | 79 | Pass |
| Shuffled-label latent control | 8 | Fail |
| Dreamer supervised head | 78 | Fail |
| Dreamer imagined reward | 28 | Fail |

Reliability required healthy 10/10 and at least 9/10 exact diagnoses in every other family. Added value additionally required Dreamer to exceed the full-history baseline by at least four correct incidents.

Paired disagreements: Dreamer alone correct **1**; full-history telemetry alone correct **0**.

| Fault family | Dreamer state / 10 | Full-history raw / 10 |
|---|---:|---:|
| Healthy | 10 | 10 |
| CPU | 10 | 10 |
| Pool | 10 | 10 |
| Delay | 10 | 10 |
| CPU + delay | 10 | 10 |
| CPU + pool | 10 | 9 |
| Pool + delay | 10 | 10 |
| Triple | 10 | 10 |

## Interpretation

No production or unseen-system reliability claim follows from this small known-regime sample. This is one training initialization and one fresh scenario batch. Mechanism supervision is part of the Dreamer candidate, so any advantage cannot be attributed solely to its world-model objectives without a matched supervised ablation. Conversely, a tie or loss does not prove all world-model approaches fail. The existing serving checkpoint was not promoted or modified.

## Provenance and execution

Training took 17.1 minutes. Collection, extraction, export and restoration are additional time. Thirteen targeted tests passed. All 160 cases passed clean measurement-window and request-success checks. Training and evaluation have no duplicate trajectories; evaluation timestamps follow readout freezing. Actual opposite-label policy invariance passed, and the evaluated checkpoint did not change.

Only the isolated inference deployment was paused for quota and restored to its original template/replicas. Fault controls were reset; temporary experiment resources were removed. All original held-out RCAEval results remain frozen.

[Protocol](LAB_FRESH_SPLIT_PROTOCOL.md) · [Verified results](evidence/lab-fresh-v1/verification.json) · [Restoration](evidence/lab-fresh-v1/restored.json)
