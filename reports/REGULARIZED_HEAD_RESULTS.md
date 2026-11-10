# Regularized head: rejected before fresh testing

**The tested regularization and PCA grid did not produce a viable world-model
classifier.** A direct-telemetry skip connection selected zero latent weight.
All 25 pure-latent settings scored 0/60 on the withheld-pair check.
No fresh incident run, deployment, checkpoint change or namespace operation was
needed. Seed 6500 remains unused.

| Selected head | Stratified validation | Withheld pair-family validation |
|---|---:|---:|
| Latent: PCA 16, penalty 0.1 | 120/140 (85.7%) | 0/60 |
| Raw telemetry: penalty 0.01 | 140/140 (100%) | 49/60 (81.7%) |
| Telemetry + latent candidate | 140/140 (100%) | 49/60 (81.7%) |

The last row selected **no latent features** and is therefore the raw classifier,
not a world-model triage improvement. The 60 stress predictions reuse incidents
from the 140 calibration records; they are not 60 additional physical incidents.

## Method

Only `latent-probe/train-features.npz` was read for model selection. All prior
held-out arrays and labels remained excluded. We used five stratified folds and
three additional splits, each withholding all examples of one pair family from
head fitting. Standardization and PCA were fitted separately inside each fold.
The frozen Dreamer itself previously trained on all 140 calibration incidents:
this is head validation, not a claim that the world model never saw those pairs.

The original grid used PCA ranks 8/16/32/64 or no PCA and averaged-loss ridge
penalties .01/.1/1/10/100. Threshold zero was fixed. Selection maximized the mean
of stratified and withheld-pair exact-set accuracy, with deterministic ties.
After seeing the poor training-only result, a documented amendment tested raw
telemetry plus PCA latent features at weights .1/1, with a weight-zero control.
This expanded search still used no held-out data. Validation is exploratory
model selection, not an unbiased estimate of final performance.

## Decision and next experiment

Reject this head before spending another live batch. The result suggests that
these linear heads do not extract a compositional diagnosis from the current
representation. It does not prove that no nonlinear readout could succeed or
that all secondary-fault information is absent: the prior latent probe recovered
some of it. No new reliability claim is supported.

Next, test **explicit mechanism supervision during world-model training**:
separate CPU/pool/delay prediction targets, supplied only as training supervision
and never as inference inputs. Validate whether this preserves mechanism signals
across combinations, alongside reconstruction/forecasting. Keep a telemetry-only
comparator and require an ablation to show that the world model adds value. A
raw-only winner must not be marketed as a successful world-model architecture.
This training change is planned, not implemented or validated in this iteration.

## Reproduction and validation

`PYTHONPATH=. python scripts/select_regularized_head.py --out <fresh-directory>`
reproduces the final amended protocol from the saved calibration arrays.
Five targeted tests passed: training-only transforms, disjoint family splits,
constant features/rank bounds, regularization shrinkage, and rejection of
uninformative latent features. The complete hybrid candidate grid was reproduced
exactly after a rank-bound robustness fix. Existing deployed code is unchanged.

See [protocol](REGULARIZED_HEAD_PROTOCOL.md), [selection artifacts](evidence/regularized-head),
and [research note](../output/pdf/dreamerv3-regularized-head.pdf).
