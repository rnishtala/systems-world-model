# Development audit of imagined diagnostic checks

Frozen before running the audit. Reuse acquisition-v1 frozen models 42/43 and only
its 60 training incidents. No held-out incident arrays or labels are loaded. This
is an in-sample mechanism audit, not confirmation, tuning, or a promotion gate.

For every incident use the fixed alphabetical inspection history, at budgets
0 through 4. At each prefix forecast every still-unseen check with 32 latent
particles, then independently reveal each candidate reading for scoring only.
Branches never alter the fixed prefix or feed future observations to prediction.
Compare imagined mean per-particle diagnosis entropy with mean posterior entropy
after that actual reading (32 particles). Decode imagined telemetry and compare
standardized queried-value MSE with a training-mean predictor. Report measurement
coordinates separately from missingness-mask coordinates. Normalization uses
training standard deviations only, flooring near-zero dimensions to one.

Also compare each candidate's predicted entropy ranking with the common ridge
classifier's realized true-class negative-log-score and accuracy after the reveal.
Softmax of ridge scores is only a ranking score, not calibrated probability.
Report candidate-pair concordance, entropy MAE/bias, selected-vs-best realized
score regret, and imagined-vs-posterior categorical Brier error. Average within
incident before reporting overall means; candidate pairs are not independent
samples. No thresholds or statistical superiority claim. All candidate records
are retained, including ties. One fixed ordering limits coverage.

Use a fresh bounded Job only in systems-world-model-lab, no network egress, no
service-account token, same quota. Temporarily pause only swm-inference and
restore its exact prior replicas/template. Keep public UI and service deployments
running. Read checkpoints without training; compare checkpoint hashes before/after.
Retain logs/results, delete experiment pod after export, and retain 24-hour Job TTL.
The next decision will depend on whether observation forecasts, entropy forecasts,
or entropy-to-diagnosis alignment fail even on training data. No model promotion.
