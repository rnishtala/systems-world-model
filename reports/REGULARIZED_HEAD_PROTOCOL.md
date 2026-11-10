# Regularized diagnosis head: prospective protocol

Keep the mechanisms-v1 seed-20 Dreamer checkpoint frozen. Use only the original
140 calibration feature arrays/labels from the frozen-state study to select a
three-output linear diagnosis head. Never load either prior held-out feature
array or predictions during selection.

Fit standardization and PCA independently inside each training fold. Candidate
latent ranks: 8, 16, 32, 64, and no PCA. Ridge penalty in the averaged squared-error
objective: .01, .1, 1, 10, 100. Classification threshold is fixed at zero. Rank 0
means no PCA. Select by the mean of five-fold stratified exact-set accuracy and
leave-one-pair-family-out accuracy (CPU+pool, CPU+delay, pool+delay). Fold seed:
7100. Ties prefer fewer dimensions, then stronger regularization. Select a raw
telemetry comparator independently using the same folds and penalties, no PCA.

These are head-validation folds: Dreamer previously trained on all calibration
incidents. They do not validate world-model generalization. Save selected models
and all candidate scores before fresh test collection. No test-driven retuning.

Fresh development test: seed 6500, 10 incidents per each of eight families,
increased load, two warmups and four frames; triples absent from fitting. Score
original Dreamer, the selected latent head and raw head on identical fresh records.
Gate: healthy 10/10, each other family at least 9/10, no invalid requests or leases.
If the latent candidate passes, require a separate prospective live confirmation
before replacing the default controller. A failed candidate is not promoted.

All work remains in this repo and systems-world-model-lab. Pause only inference;
runner uses 2.4 CPU/4 GiB, applications retain 1.6 CPU within the existing quota.
No Dreamer retraining, external namespace changes, LLM or remediation is included.
Restore prior inference after success or failure. Hash checkpoint before/after.

## Training-only amendment before any fresh evaluation

The pure-latent grid failed the withheld-combination check (chosen candidate:
0/60). Do not collect fresh data for that candidate alone. Test a telemetry skip
connection: concatenate the eight raw features with 8/16/32 PCA latent features,
standardized within folds, using latent weights .1 or 1. Include weight-zero
(raw-only) as a candidate. Use the same five penalties and selection objective;
ties prefer no latent features, then fewer latent features, lower latent weight,
stronger regularization. All transformations and selection remain training-only.
Preserve the original pure-latent result. If raw-only wins, explicitly report
that the world model did not improve this head; do not call it world-model triage.
