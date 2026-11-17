# Partial-telemetry matched-objective results — 16 November 2026

**No added diagnostic benefit from the world-model objective was demonstrated. The matched diagnosis-only recurrent control tied Dreamer in one seed and exceeded it by one case in the other. The primary exploratory gate was not met.**

New RCAEval RE2-SS (Sock Shop) collection: 60 training incidents and 30 held-out incidents. All 90 passed the prespecified timestamp checks. No test answers were uploaded. All predictions were frozen before a single local scoring pass.

## Held-out service localization

| Condition | Method | Seed 42 top1 | Seed 43 top1 | Seed 42 top3 | Seed 43 top3 |
|---|---|---:|---:|---:|---:|
| partial | World-model + diagnosis | 30/30 | 28/30 | 30/30 | 30/30 |
| partial | Matched recurrent diagnosis only | 30/30 | 29/30 | 30/30 | 30/30 |
| partial | Raw telemetry ridge | 28/30 | 28/30 | 29/30 | 29/30 |
| full | World-model + diagnosis | 29/30 | 29/30 | 30/30 | 30/30 |
| full | Matched recurrent diagnosis only | 30/30 | 30/30 | 30/30 | 30/30 |
| full | Raw telemetry ridge | 29/30 | 29/30 | 30/30 | 30/30 |

Both seeds use the same 30 test incidents; they are not 60 independent observations. Raw ridge is deterministic and its repeated rows are one result.

## What was controlled

Both trained candidates use the same DreamerV3 size1m stochastic recurrent architecture, diagnosis head, seeds, step budget, replay configuration, masked inputs and fixed ridge100 readout. The control optimizes only diagnosis cross entropy. The candidate also optimizes the original upstream loss, including reconstruction, dynamics/KL, reward, continuation, actor and critic. This isolates the combined upstream training objective, not dynamics alone. Unused control heads remain allocated.

Primary inputs hide 8 of 16 service groups independently at each post-onset frame, using a fixed label-independent hash. Both values and presence masks are zeroed. All pre-onset telemetry remains available. The secondary full-input condition uses the same checkpoints with complete observations and a readout fitted to complete training inputs; it is not a separately trained full-observation model.

The model learns from the masked observations themselves. It never receives the hidden post-onset telemetry as a reconstruction target. This experiment tests intermittent-observation robustness, not a learned inspection policy or prediction of remediation outcomes.

## Paired comparisons

| Seed/condition/comparator | Both correct | World-model only | Comparator only | Both wrong |
|---|---:|---:|---:|---:|
| 42-partial-control | 30 | 0 | 0 | 0 |
| 42-partial-raw | 28 | 2 | 0 | 0 |
| 43-partial-control | 28 | 0 | 1 | 1 |
| 43-partial-raw | 27 | 1 | 1 | 1 |
| 42-full-control | 29 | 0 | 1 | 0 |
| 42-full-raw | 28 | 1 | 1 | 0 |
| 43-full-control | 29 | 0 | 1 | 0 |
| 43-full-raw | 28 | 1 | 1 | 0 |

The primary gate required an improvement of at least 10 percentage points over both the matched control and raw ridge in each seed. It is a prespecified exploratory threshold, not statistical proof or a deployment acceptance gate.

## Per-fault primary top1

| Method/seed | CPU | Delay | Disk | Loss | Memory | Socket |
|---|---:|---:|---:|---:|---:|---:|
| World-model/42 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| World-model/43 | 5/5 | 5/5 | 4/5 | 5/5 | 4/5 | 5/5 |
| Control/42 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Control/43 | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 |
| Raw/42 | 5/5 | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 |
| Raw/43 | 5/5 | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 |

## Training and audits

| Model | Training/extraction seconds | Training metric rows audited | Base-loss scalar derivative |
|---|---:|---:|---:|
| world/42 | 328.3 | 10 | 1.0 |
| control/42 | 273.3 | 8 | 0.0 |
| world/43 | 323.1 | 10 | 1.0 |
| control/43 | 270.8 | 8 | 0.0 |

All four models passed target-isolation audits. Actual training metrics verify that optimized loss equals diagnosis loss for the controls. The loss-combination derivative is 0 for controls and 1 for candidates; this is a scalar-objective check, not an exhaustive per-parameter gradient audit. Twelve focused local checks passed.

## Scope and operations

These are fresh incidents in a different benchmark application, but models train on that application; no zero-shot system transfer is claimed. Known fault onset and five minutes of post-onset observations remain available. Healthy cases, unknown fault families, causal explanation, adaptive investigation and automated repair are not tested. Results are not tuned after scoring.

All work stayed in systems-world-model-lab under existing quota and network restrictions. The inference deployment was restored with its original template. Five services are healthy, no experiment pods remain, checkpoints are preserved, and no candidate was promoted.

[Frozen protocol](RCAEVAL_PARTIAL_PROTOCOL.md) · [Scores and errors](evidence/rcaeval-partial-v1/scores.json) · [Prediction hashes](evidence/rcaeval-partial-v1/frozen-hashes.json) · [Input audit](evidence/rcaeval-partial-v1-source/mask-audit.json) · [Restoration](evidence/rcaeval-partial-v1/restored.json)

## Decision and next useful question

The recurrent diagnosis-only control improved on raw ridge by two cases in seed 42 and one case in seed 43 under partial telemetry. Dreamer improved by two and zero cases, respectively. These small results support further study of recurrent diagnosis representations, but do not establish an advantage from the additional world-model training objectives. No case was uniquely corrected by Dreamer relative to the matched control in either primary comparison.

The raw baseline was already 28/30, leaving only two cases of headroom; even a perfect model could not meet the prespecified three-case advantage over raw telemetry. The direct matched-control comparison is therefore more informative than the gate alone. No threshold was changed after scoring.

Do not tune these 30 test incidents or promote a candidate. The next useful experiment should measure whether predictions help choose the next diagnostic observation, reducing checks or time at the same diagnosis accuracy. That requires a fresh action/observation evaluation with equal observation budgets for all methods; this static replay cannot validate adaptive investigation or remediation. Preserve this negative result rather than treating good absolute accuracy as proof of world-model value.
