# Partial-evidence diagnosis: development comparison

**Stronger readouts improve partial diagnosis, but do not justify another live run.**
On matched nested cross-validation of the 80 development incidents, Dreamer-state
two-check diagnosis improved from 56/80 with fixed ridge to 62/80 with a selected
linear or nonlinear readout. Reliable stopping still required 2.9875 of 3 checks.
The prespecified development gate failed; no new fault injection, deployment,
model training or serving change was made.

## Matched development results

Each incident is scored once in an outer held-out fold. Readout selection and
stopping calibration use only that fold's training partition. Both representations
receive the same candidate models, tuning budget and data splits.

| Readout | 0 checks | 1 check | 2 checks | 3 checks |
|---|---:|---:|---:|---:|
| Dreamer, fixed ridge | 19/80 | 34/80 | 56/80 | 80/80 |
| Dreamer, tuned readout | 25/80 | 43/80 | 62/80 | 80/80 |
| Telemetry, fixed ridge | 25/80 | 39/80 | 58/80 | 79/80 |
| Telemetry, tuned readout | 35/80 | 36/80 | 55/80 | 80/80 |

| Selected stopping policy | Correct | Mean checks | Reliability gate |
|---|---:|---:|---|
| Dreamer state | 80/80 | 2.9875 | Pass |
| Telemetry history | 80/80 | 3.0000 | Pass |

Dreamer saves one check across the entire 80-case batch, far below the required
0.5 average checks. Four of five telemetry calibration folds found no eligible
threshold and used the conservative all-check fallback. Hyperparameter tuning
also reduced telemetry accuracy at some partial budgets, illustrating instability
with this small development sample.

## What limits early stopping

The original frozen readout's development audit shows:

- After CPU inspection, Dreamer makes no CPU-bit errors, but still makes 34 pool
  and 30 delay-bit errors. Telemetry similarly has no CPU-bit errors.
- After CPU and pool inspection, both original readouts get those two mechanism
  bits right in every case. Their remaining errors are entirely the delay bit:
  24 for Dreamer and 19 for telemetry.
- The tuned Dreamer readout reduces two-check delay errors to 18, with zero CPU
  or pool-bit errors. The trace inspection that directly exposes delay evidence
  is still last in the fixed ordering.
- Original Dreamer scores can be confidently wrong: ten zero-check development
  errors have minimum absolute signed score >=1. These scores are not calibrated
  probabilities; an arbitrary margin is not a reliability guarantee.

This points to unresolved evidence about delay, rather than only a stopping
threshold problem. It does not prove delay is impossible to infer earlier, or
that the representation cannot support a better decision rule.

## Next useful experiment

Compare check orders on development data, especially moving trace inspection
earlier, before spending another fresh live batch. Give a telemetry policy the
same options and costs. A useful world-model result must beat that baseline in
reliable diagnoses per check; merely beating the previous fixed order is not
enough. Preserve all completed evaluations, freeze any promising policy, and then
confirm it on new incidents. Do not tune further on these outer results while
presenting them as an unchanged validation set.

## Scope and verification

This is development evidence using scenario 8201 only. It does not replace or
update the preceding scenario 8202 evaluation results. The fixed and tuned
readouts here train on 64 development incidents per outer fold; comparisons with
the earlier original-training readouts are descriptive, not matched experiments.
Dreamer seed 51 remains frozen and was trained on a separate batch. This compares
supervised readouts of its state, not native Dreamer planning or the isolated
benefit of world-model training objectives. No fitted candidate was promoted.

Ten targeted tests passed, including a test that changing outer-held-out labels
cannot change that fold's predictions, selected models or stopping threshold.
Verification reproduced every score and threshold, checked disjoint nested splits
and input hashes, and confirmed only the three allowlisted development inputs
were used. The comparison ran locally without accessing Kubernetes.

[Protocol](LAB_PARTIAL_READOUT_PROTOCOL.md) · [Detailed results](evidence/lab-partial-readout-v1/summary.json) · [Verification](evidence/lab-partial-readout-v1/verification.json)
