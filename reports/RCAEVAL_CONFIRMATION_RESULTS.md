# RCAEval fresh-collection confirmation — 16 November 2026

**Outcome: Dreamer did not improve on either simpler comparator on this held-out collection. One seed tied them; the other missed three packet-loss cases.**

The fixed training recipe was evaluated on a separate RCAEval collection (RE2-OB): 60 training incidents and 30 untouched test incidents. No exclusions were needed. All predictions were frozen and hashed before local scoring. Test labels were not uploaded.

| Method | Seed 42 top-1 | Seed 43 top-1 | Seed 42 top-3 | Seed 43 top-3 |
|---|---:|---:|---:|---:|
| Diagnosis-supervised Dreamer | 30/30 | 27/30 | 30/30 | 30/30 |
| Supervised temporal MLP | 30/30 | 30/30 | 30/30 | 30/30 |
| Raw sequence ridge | 30/30 | 30/30 | 30/30 | 30/30 |

## Paired comparisons

Counts compare the same 30 test incidents; the two seeds are not independent test sets.

| Comparator | Both correct | Dreamer only | Comparator only | Both wrong |
|---|---:|---:|---:|---:|
| baseline-42/predictions.json | 30 | 0 | 0 | 0 |
| readout-42/raw-predictions.json | 30 | 0 | 0 | 0 |
| baseline-43/predictions.json | 27 | 0 | 3 | 0 |
| readout-43/raw-predictions.json | 27 | 0 | 3 | 0 |

## Per-fault top-1

| Method/seed | CPU | Delay | Disk | Loss | Memory | Socket |
|---|---:|---:|---:|---:|---:|---:|
| Diagnosis-supervised Dreamer / 42 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Diagnosis-supervised Dreamer / 43 | 5/5 | 5/5 | 5/5 | 2/5 | 5/5 | 5/5 |
| Supervised temporal MLP / 42 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Supervised temporal MLP / 43 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Raw sequence ridge / 42 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Raw sequence ridge / 43 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

## Interpretation and limits

This is transfer of a training recipe to a fresh collection of Online Boutique, with new weights trained on that collection. It is not transfer of the RE1 checkpoint, unseen-system generalization, or zero-shot diagnosis. Labels identify the responsible service, not a demonstrated causal explanation.

The neural comparator uses the same incident labels and ordered telemetry, without a dynamics or reconstruction objective. Its architecture, capacity, and optimizer budget differ from Dreamer. This is a practical comparator, not a causal ablation of the dynamics loss. Dreamer runs 32,000 environment steps; the MLP runs 32,000 minibatch optimizer updates. Those numbers are not equivalent compute budgets.

Inputs cover five minutes before and after known fault onset. One training incident ends at 209 seconds after onset; all test incidents cover the full window. Healthy cases, unknown faults, investigation policies and remediation are not evaluated. The primary Dreamer readout was fixed before results: pooled post-onset full state, 16 posterior particles, ridge100. No confirmation readout search was performed.

## Operational verification

Nine focused checks passed. Actual trained-model label-isolation audits are retained for both seeds. All work stayed in `systems-world-model-lab` under the existing quota. The inference deployment was temporarily paused and restored with the same pod template; no candidate was promoted.

[Frozen protocol](RCAEVAL_CONFIRMATION_PROTOCOL.md) · [Scores](evidence/rcaeval-confirm-v1/scores.json) · [Prediction hashes](evidence/rcaeval-confirm-v1/frozen-hashes.json) · [Input audit](evidence/rcaeval-confirm-v1-source/input-audit.json)

## Run times

| Seed | Dreamer train/extract | Fixed readout | Supervised MLP |
|---|---:|---:|---:|
| 42 | 341.0s | 17.8s | 69.3s |
| 43 | 344.5s | 17.9s | 68.0s |

## Error review and next experiment

Seed43 confused currencyservice with productcatalogservice, productcatalogservice with currencyservice, and recommendationservice with productcatalogservice on the three corresponding packet-loss incidents. The correct service ranked second in all three. Seed42 and both comparators localized every incident correctly. No test-driven tuning was performed after this result.

This collection does not establish added diagnostic value from a world model. The primary Dreamer pipeline took about359–362 seconds per seed including extraction/readout, versus68–69 seconds for the supervised MLP, with shared preprocessing excluded from the MLP timer. These are measured end-to-end method timings, not a controlled compute-efficiency benchmark. Raw ridge is deterministic; its duplicate rows across seeds are one baseline result, not separate confirmations.

Keep these results frozen. A useful next experiment would prespecify partial telemetry, changing load, or sequential investigation where predicting system evolution may help, and compare against a matched recurrent supervised encoder. Use fresh evaluation incidents; do not tune on these 30 cases. No serving change is justified by this benchmark.
