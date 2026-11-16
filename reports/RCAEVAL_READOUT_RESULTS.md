# RCAEval validation follow-up — 16 November 2026

**Outcome: better readout and longer training improved validation accuracy, but neither seed beat raw telemetry. No model was promoted.**

Development-only results: all original test predictions and scores remain frozen.

| Model | Validation top-1 | Validation top-3 |
|---|---:|---:|
| Original 8k, single-particle final full state | 5/25 | 13/25 |
| 8k, pooled full state /16 particles | 14/25 | 23/25 |
| 32k seed 42, same pooled readout | 16/25 | 22/25 |
| 32k seed 43, same pooled readout | 19/25 | 24/25 |
| Raw telemetry, ridge1 | 23/25 | 25/25 |
| Missingness only, ridge100 | 8/25 | 19/25 |

The pooled full-state readout with ridge100 was selected from the frozen-model validation study and used as the primary longer-training comparison. These are the same 25 development incidents across seeds; they are not 50 independent cases. No test case was loaded.

## Complete longer-training readout menu

| Readout | Seed42 top-1 | Seed43 top-1 |
|---|---:|---:|
| last_deter_ridge1 | 14/25 | 15/25 |
| last_deter_ridge1_shuffled | 5/25 | 5/25 |
| last_deter_ridge100 | 10/25 | 12/25 |
| last_deter_ridge100_shuffled | 3/25 | 9/25 |
| post_mean_deter_ridge1 | 10/25 | 16/25 |
| post_mean_deter_ridge1_shuffled | 5/25 | 5/25 |
| post_mean_deter_ridge100 | 9/25 | 13/25 |
| post_mean_deter_ridge100_shuffled | 4/25 | 6/25 |
| post_mean_full_ridge1 | 12/25 | 17/25 |
| post_mean_full_ridge1_shuffled | 3/25 | 3/25 |
| post_mean_full_ridge100 | 16/25 | 19/25 |
| post_mean_full_ridge100_shuffled | 5/25 | 6/25 |
| raw_ridge1 | 23/25 | 23/25 |
| raw_ridge1_shuffled | 9/25 | 9/25 |
| raw_ridge100 | 23/25 | 23/25 |
| raw_ridge100_shuffled | 7/25 | 7/25 |
| mask_only_ridge1 | 7/25 | 7/25 |
| mask_only_ridge1_shuffled | 3/25 | 3/25 |
| mask_only_ridge100 | 8/25 | 8/25 |
| mask_only_ridge100_shuffled | 4/25 | 4/25 |

## Interpretation and boundaries

The saved short-run model contains more usable information than its original single-particle final-state readout showed: changing only extraction/readout increased validation accuracy from 5/25 to 14/25. Stronger regularization and posterior/temporal averaging were changed together, so this study does not isolate which factor caused the improvement.

Missingness-only performance of 7–8/25 does not explain the raw model’s 23/25. It does not rule out other dataset shortcuts. Shuffled-label scores are retained as controls, not candidates for selection.

Training uses only 73 labeled training incidents for classifier fitting. World-model fitting uses their telemetry with zero rewards and one no-op action. Both 32k models are fresh training runs, with unchanged normalization and losses. This is passive representation learning, not remediation planning.

The upstream normalized exploration statistic divides by a zero entropy range in a one-action environment. This explains its NaN logging value; it is not the model’s diagnosis output. Readout code rejects nonfinite features and scores.

All work ran in systems-world-model-lab with no quota increase or application fault injection. The inference deployment was temporarily paused to fit the existing budget; its exact template and replica count were restored. Existing serving weights were not changed.

[Protocol](RCAEVAL_READOUT_PROTOCOL.md) · [Short readout evidence](evidence/rcaeval-readout-v1/summary.json) · [Seed42](evidence/rcaeval-long-v1/readout-42-summary.json) · [Seed43](evidence/rcaeval-long-v1/readout-43-summary.json)

## Next step

Test an auxiliary root-cause supervision objective during world-model training, while retaining Dreamer’s dynamics/reconstruction objectives and the raw-telemetry comparator. More reconstruction training alone has not closed the gap. Use validation for development and a fresh prespecified holdout for confirmation; do not reuse the original test set for model selection.

Five focused tests passed. Training plus original feature extraction took 332.0 and 340.7 seconds for seeds 42 and 43; pooled readouts added about 16 seconds each. All five lab deployments are healthy, and no benchmark pods remain.
