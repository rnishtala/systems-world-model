# RCAEval diagnosis-supervision results — 16 November 2026

**Outcome: the prespecified supervised Dreamer readout scored 25/25 for both seeds, versus 16/25 and 19/25 without supervision and 23/25 for raw telemetry. No model was promoted.**

These are development validation results on the same 25 incidents. The original test set was not loaded or rescored.

| Method | Seed 42 top-1 | Seed 43 top-1 |
|---|---:|---:|
| Unsupervised Dreamer, 32k steps | 16/25 | 19/25 |
| Diagnosis-supervised Dreamer, 32k steps | 25/25 | 25/25 |
| Raw-telemetry classifier | 23/25 | 23/25 |

Primary readout remained the prespecified post-onset mean full state, 16 posterior particles, ridge penalty 100. Both models trained from scratch for 32,000 steps with the same seeds, data, architecture and normalization as their unsupervised comparators. The only training-objective addition was post-onset service cross-entropy with weight 1. Existing Dreamer dynamics/reconstruction objectives remained enabled.

## Label isolation and scope

Both actual trained models passed all four audits: target excluded from encoder and decoder, auxiliary head included in the optimizer, and identical policy outputs under opposite supplied labels with a fixed RNG key. All inference calls used dummy zeros and stripped the target before the policy. Only training labels contributed to optimization; validation labels were used for scoring and development comparisons.

The shuffled-label controls below shuffle readout training labels only. The supervised world model was trained with the correct training labels; these are not end-to-end shuffled-supervision controls. This remains passive telemetry diagnosis with known onset and five minutes of post-onset evidence, not remediation planning or unseen-system confirmation.

## Secondary readouts

| Readout | Seed 42 top-1 | Seed 43 top-1 |
|---|---:|---:|
| last_deter_ridge1 | 25/25 | 24/25 |
| last_deter_ridge1_shuffled | 4/25 | 5/25 |
| last_deter_ridge100 | 25/25 | 23/25 |
| last_deter_ridge100_shuffled | 13/25 | 14/25 |
| post_mean_deter_ridge1 | 25/25 | 25/25 |
| post_mean_deter_ridge1_shuffled | 0/25 | 2/25 |
| post_mean_deter_ridge100 | 25/25 | 25/25 |
| post_mean_deter_ridge100_shuffled | 7/25 | 4/25 |
| post_mean_full_ridge1 | 25/25 | 24/25 |
| post_mean_full_ridge1_shuffled | 2/25 | 6/25 |
| post_mean_full_ridge100 | 25/25 | 25/25 |
| post_mean_full_ridge100_shuffled | 4/25 | 5/25 |
| raw_ridge1 | 23/25 | 23/25 |
| raw_ridge1_shuffled | 9/25 | 9/25 |
| raw_ridge100 | 23/25 | 23/25 |
| raw_ridge100_shuffled | 7/25 | 7/25 |
| mask_only_ridge1 | 7/25 | 7/25 |
| mask_only_ridge1_shuffled | 3/25 | 3/25 |
| mask_only_ridge100 | 8/25 | 8/25 |
| mask_only_ridge100_shuffled | 4/25 | 4/25 |

Six focused tests passed, including target stripping and test-input exclusion. No candidate was promoted. All cluster work stayed in `systems-world-model-lab`, under the existing quota and network policies. The lab inference deployment was restored; the serving package was not modified.

[Protocol](RCAEVAL_SUPERVISION_PROTOCOL.md) · [Seed 42 evidence](evidence/rcaeval-supervised-v1/readout-42-summary.json) · [Seed 43 evidence](evidence/rcaeval-supervised-v1/readout-43-summary.json) · [Label audit](evidence/rcaeval-supervised-v1/label-audit-42.json)

## What improved and what remains

Both seeds corrected the raw classifier’s two validation errors: packet loss in
`adservice` and CPU stress in `currencyservice`, each incorrectly attributed by
the raw classifier to `checkoutservice`. This is root-cause classification
evidence, not proof of a learned causal explanation.

The two seeds use the same 25 validation cases; they are not 50 independent
incidents. This suite tests service localization after a known fault onset, not
healthy-case false alarms, unknown faults or automated repair. Validation has
been reused for development.

Next: freeze the configuration and evaluate on a fresh prespecified benchmark
holdout, including an equally supervised non-world-model baseline. That control
is needed to separate the benefit of supervised representation learning from
the contribution of Dreamer’s dynamics/reconstruction objectives. Keep the
original test scores frozen and do not promote on this development result.
