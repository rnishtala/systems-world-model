# Imagined check audit — 17 November 2026

**Finding: the acquisition score is poorly aligned with diagnostic usefulness,
even on training incidents. No new confirmation or model promotion.**

We reused both frozen Dreamer models on the 60 acquisition-training incidents.
For each fixed alphabetical prefix, we imagined every remaining check, then
independently revealed each actual reading for scoring: 900 candidates per model,
1,800 total. The original 30-case test set was not loaded or changed. Checkpoint
hashes match the original acquisition experiment, and no weights were updated.

| Measure | Seed 42 | Seed 43 |
|---|---:|---:|
| Pairwise agreement: imagined vs actual posterior entropy | 59.3% | 53.1% |
| Pairwise agreement: imagined entropy vs shared diagnosis score | 52.8% | 49.3% |
| Entropy forecast MAE (nats) | 0.231 | 0.215 |
| Entropy bias, imagined minus posterior (nats) | +0.191 | +0.179 |
| Query measurement forecast standardized MSE | 920.88 | 1539.50 |
| Training-mean measurement baseline MSE | 0.917 | 0.917 |
| Selected next check diagnosis accuracy | 82.7% | 82.0% |
| Best actual next check accuracy, hindsight only | 99.3% | 99.3% |

The measurement-error means are highly skewed: median candidate MSE was 0.935 and
0.919, with maxima 21,163 and 22,247. Normalizing by small training standard
deviations can amplify errors; these averages alone do not identify a physical
telemetry failure. A per-coordinate scale audit is required before attributing
the extreme values to a particular modeling defect. Mask forecast MSE was about
0.214 for both seeds, versus 0.125 for the training-mean baseline.

The more directly actionable finding is weak check ranking. Positive entropy
bias means imagined next states were less confident on average than actual
observed next states; this is not evidence of general overconfidence. Pairwise
agreement with the shared classifier's realized true-class score was approximately
chance. This supports investigating the mismatch between the learned diagnosis
head used for acquisition and the shared ridge classifier used for final answers.
It does not establish that head mismatch is the sole cause: the imagined-state
forecasts themselves also show deficiencies.

Selected/best accuracy is averaged over all five prefix budgets, on training
cases, with hypothetical candidate outcomes. The hindsight best check is an
oracle diagnostic bound, never an executable policy. These are not held-out
acquisition curves and must not be compared directly to the previous 30-case
confirmation scores. Ridge softmax scores are not calibrated probabilities.
Incident averages are the reporting units; candidates and particle samples are
not independent incidents. A single fixed prefix order limits coverage.

## Next useful experiment

First inspect decoded errors per telemetry coordinate, then test a matched
acquisition score: apply the same shared diagnostic classifier to imagined
observations, versus the current auxiliary-head entropy. Select changes using
only development data, including a development validation split or cross-fitting;
training results alone cannot establish generalization. Only after development
improvement should a fresh untouched collection be used for confirmation. More
training or another test run without fixing this alignment is not justified yet.

## Execution and isolation

The Job ran for 46 seconds (about 20 seconds of audit per seed); staging, export,
and restoration add wall time. Seven observation/scoring tests passed. Exported
hashes and unchanged checkpoints were verified. The inference deployment's exact
template and original replica count were restored. The experiment Job and reader
Pod were removed; public UI and lab services remained running. No other namespace,
AutoSRE code, held-out results, or serving checkpoint was modified.

[Protocol](RCAEVAL_FORECAST_AUDIT_PROTOCOL.md) ·
[Aggregate evidence](evidence/rcaeval-forecast-audit-v1/aggregate.json) ·
[Restoration](evidence/rcaeval-forecast-audit-v1/restored.json)
