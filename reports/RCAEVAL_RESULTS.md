# RCAEval pilot results — 16 November 2026

**Outcome: benchmark execution completed; this Dreamer configuration did not improve triage.**

All cluster work ran in `systems-world-model-lab`. Five existing deployments are healthy, the inference template and replica count were restored, and no serving checkpoint was replaced. Experiment pods and jobs were removed; model outputs remain at `/results/rcaeval-v1/dreamer` on the lab PVC.

## Held-out service localization

| Method | Test top-1 | Test top-3 | Test Avg@5 | Validation top-1 |
|---|---:|---:|---:|---:|
| baro_style | 6/25 | 19/25 | 0.680 | 5/25 |
| raw_sequence | 24/25 | 25/25 | 0.992 | 23/25 |
| dreamer_state | 5/25 | 16/25 | 0.536 | 5/25 |

The two supervised classifiers each achieved 73/73 on training incidents. Dreamer’s strong training fit and poor held-out results are consistent with overfitting or an unhelpful short-run representation. They do not establish that world models cannot help. The raw-sequence classifier is the stronger comparator here, although dataset shortcuts and performance on new systems remain untested; the BARO-style baseline is unsupervised and uses different preprocessing.

## What ran

- Pinned RCAEval RE1-OB: 125 source incidents; two excluded before scoring because their published injection times lie outside available telemetry. Both exclusions affect training, leaving 73 training, 25 validation and 25 test cases.
- Fixed whole-incident repetition split, anonymized input paths, separate scorer labels. No identical telemetry arrays across incidents. Known onset and five minutes of post-onset evidence for every method.
- Actual upstream DreamerV3, initialized from scratch, size1m, seed42, 8,000 steps. Twenty metric frames with missing-data masks. A single no-op action and zero rewards make this passive representation learning, not intervention-policy learning.
- Ridge classifiers use the same training service labels and fixed penalty1. No tuning after viewing validation/test results.
- Baseline compute: 6.49 seconds. Dreamer training plus compilation: 93.26 seconds. Training, state extraction and readouts: 94.79 seconds. These are compute times, not incident-to-diagnosis latency; the observation budget is five minutes.
- Two focused adapter/scoring tests passed. Training and job completion succeeded. The upstream `train/rand/action` diagnostic is NaN for the one-choice action space; policy loss is zero. The saved classifier parameters were separately verified finite. The one-choice exploration statistic should be checked before extending training; it is not an accuracy result.
- CPU cap2, memory cap4Gi, unchanged namespace quota, no pod network access, no service-account token, no live fault injection. Only the lab inference pod was briefly paused to fit its existing quota.

## Next experiment

Use validation-only comparisons to investigate longer training, deterministic-state or pooled temporal readouts, and a smaller classifier. Add repeated seeds and a shuffled-label control. Keep this test result frozen; after development, use a fresh prespecified dataset/system holdout for confirmation. A second benchmark should include the official BARO preprocessing and evaluator before making paper-comparable claims. Do not promote this model.

[Protocol](RCAEVAL_PROTOCOL.md) · [Runbook](../docs/RCAEVAL_RUNBOOK.md) · [Scores](evidence/rcaeval-v1/scores.json) · [Dataset provenance and exclusions](evidence/rcaeval-v1/provenance.json) · [Restore verification](evidence/rcaeval-v1/restored.json)

Sources: [RCAEval paper](https://arxiv.org/abs/2412.17015), [dataset](https://huggingface.co/datasets/phamquiluan/RCAEval), [benchmark implementation](https://github.com/phamquiluan/RCAEval). The local robust-shift baseline is explicitly BARO-style, not a reproduction of the published benchmark numbers.
