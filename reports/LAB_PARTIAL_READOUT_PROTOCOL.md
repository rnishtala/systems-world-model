# Development-only partial-observation readout experiment

Freeze this protocol before running the comparison. Reuse only scenario 8201's
80 development incidents and their saved prefix features. Do not load scenario
8202 or 8102 evaluation trajectories, features or predictions. Dreamer seed 51 is
frozen and was trained on a different incident batch (8101). No cluster changes,
new fault injection or serving promotion during this development comparison.

Audit the original readouts' development errors by check budget, mechanism bit,
and score margin. Signed ridge scores are not probabilities. Compare each method
on identical stratified five-fold outer splits (seed 8301), with two incidents per
fault family per held-out fold. Each incident is predicted exactly once by a
readout and stopping rule that did not use its label or features for fitting.

For each outer fold, use four-fold inner validation within its 64 training cases
(seed 8302 + fold) to select a readout independently at each prefix. Candidate
order (also deterministic accuracy-tie priority): linear ridge penalties
1000, 100, 10, 1; RBF kernel ridge gamma 0.1 or 1 and penalty 0.01 or 0.1.
Normalize features using fitting cases only; RBF distance is squared standardized
distance divided by feature count. Fit signed mechanism targets with a mean-label
intercept. Select by exact fault-set accuracy on inner out-of-fold predictions.
Apply the identical candidate set and tuning budget to raw history and latent state.
Compare against fixed penalty-1 linear ridge trained on the same outer folds.

Calibrate each outer fold's stopping threshold using only its selected inner
out-of-fold scores: grid [0, .25, .5, 1, 2, 4, infinity], first prefix with minimum
absolute bit score >= threshold, otherwise three checks. Eligibility requires all
healthy correct, >=90% in every other family (rounded up), and at most one fewer
correct case than full-check diagnosis. Minimize mean checks, tie toward the
higher threshold; default to all checks if no eligible threshold exists. Inner
scores also drive model selection, so their apparent quality is optimistic;
only the outer predictions are used for the development gate.

Report aggregate outer-fold fixed-budget accuracy, stopping accuracy, per-family
counts, mean checks, calibration failures, and selected models. No tuning after
seeing outer results. Development gate for spending another live run: latent
stopping gets healthy 10/10, every other family >=9/10, loses <=1 case against its
own full-check predictor, and saves >=0.5 average checks versus both full checking
and the tuned telemetry stopping policy. Passing is only a reason to freeze a
candidate and collect a new confirmation batch, not a validation claim. If it
fails, stop and document what was learned without running another live batch.

This tests supervised readouts of a frozen representation, not native Dreamer
planning or the causal value of world-model training. Existing development data
has informed the choice of question; nested validation is development evidence,
not an untouched external test. Fixed CPU/pool/trace order and elapsed-observation
confounding remain. Preserve input hashes, splits, predictions, candidates and
results. Do not overwrite earlier evidence.
