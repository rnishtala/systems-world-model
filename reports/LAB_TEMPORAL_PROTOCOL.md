# Trace-only temporal diagnosis under held-out load

2026-11-18. Freeze before collecting data or training.

Question: does a trained Dreamer representation improve diagnosis when direct CPU,
pool-wait and database-duration measurements are unavailable, and load changes over
time? This is a representation/generalization experiment, not an action-planning
or early-stopping claim. Architecture and model size remain unchanged.

Use only systems-world-model-lab, existing bounded application controls, and a new
/results/lab-temporal-v1 directory. No previous confirmation data enter training.
Collect 80 training incidents (seed 8601), ten per known family. Use the usual
unshifted fault severities and alternating concurrent load [4,6,4,6] or [6,4,6,4],
with phase balanced within each family. Collect 80 fresh test incidents (seed 8602)
only after fitting every model/readout. Test loads [7,9,7,9] or [9,7,9,7]; severity
ranges remain identical to training, isolating workload transfer.

For each incident reset controls, warm up twice, then reset metrics before each
completed batch so every frame measures one window. Record all raw metrics for
audit, but expose only checkout latency/error/load, payment self latency, ledger
latency, and payment downstream latency. Never expose controls, seed, family,
CPU, pool-wait or DB timings to any classifier. Faults persist through four frames.
All faults lease for 45 seconds; reject/preserve incidents exceeding 35 seconds or
with request/count failures. No exclusions or rerolls. Reset in finally.

Train one new supervised upstream DreamerV3 seed 61 for 100,000 steps from scratch
on the empirical training trajectories. A new 14-dimensional trace-only view
retains the existing architecture; supervised targets are still excluded from the
encoder and decoder. Training guided trajectories wait through the sequence before
diagnosing. All methods get the same four frames and no diagnosis-correctness
reward during evaluation. Record actual label-invariance and checkpoint hashes.

Compare frozen final-state Dreamer readout, flattened telemetry history, and latest
telemetry frame. Select each readout using the existing eight linear/RBF candidates
and five stratified folds (seed 8603) on training data only, exact-diagnosis accuracy,
first candidate on ties. Dreamer embeddings are trained on the whole training set;
these folds select readouts only and are NOT independent world-model validation.
Freeze fitted readouts before collecting test data. Also report auxiliary Dreamer
head and shuffled-label latent control. No test-driven tuning or retraining.

Primary outcomes: exact diagnosis and per-family accuracy on the 80 held-out-load
incidents, and paired Dreamer versus full-history telemetry wins/losses. Exploratory
added-value gate: all healthy correct, >=9/10 each other family, and >=4 additional
correct diagnoses versus the stronger of history/latest baselines. Report the paired
exact test and uncertainty; one seed/small controlled lab cannot establish broad
superiority. No automatic serving promotion. Restore the existing inference deployment
and clean experiment Job/stager after collecting evidence, including on failure.
