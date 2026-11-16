# RCAEval external telemetry pilot

Namespace: `systems-world-model-lab`. No other namespace is used. No live fault
injection or application network access. Existing quota remains unchanged.

Pin dataset revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`.
RE1-OB: 125 source cases, two timestamp-invalid cases excluded before scoring
(see evidence provenance). Repetitions 1–3 train (73), 4 validation (25), 5 test
(25). Split whole incidents; do not split overlapping windows. Ground truth and
original case names are absent from blind arrays. Only training service labels
are uploaded to training; held-out labels stay outside the cluster job.

Each case uses [-300,+300) seconds around the supplied onset. This is diagnosis
with known onset and five minutes of post-onset data, not anomaly detection.
Dreamer receives 20 thirty-second metric means, normalized using that case's
pre-onset data, plus observation masks. Metric vocabulary is training-only.
Missing data maps to zero with mask zero. Scaling is clipped at +/-20.

Train upstream DreamerV3 revision e01491fad6434b2245a3b8ca201dd7faedcc458c,
size1m, seed42, 8000 steps, one no-op action, zero rewards. This tests passive
recurrent representation learning; it does not demonstrate intervention planning.
A ridge service classifier (penalty1) uses training labels only, over the final
Dreamer state. Compare a ridge classifier on the same flattened telemetry
sequence and a BARO-style robust-shift baseline on raw metrics. The latter is
not an exact reproduction of RCAEval's published preprocessing/results.
No hyperparameter selection using validation or test. Report both, but test is
primary. Top-1/top-3 service accuracy and Avg@5; report exclusions, per-fault
counts, training/runtime and fixed five-minute evidence budget. Single-seed pilot
results cannot establish generalization to unseen systems or fault families.

The inference deployment may be paused during bounded training to fit existing
quota; restore its original replica count afterward. Training deadline one hour;
no automatic retries. Preserve raw predictions and model outputs on lab PVC.
