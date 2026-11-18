# Fresh lab train/evaluation comparison

Frozen before collection. All work stays in systems-world-model-lab. No historical
calibration, RCAEval incident, or prior evaluation case is reused for training.

Collect 80 new training incidents with seed 8101, ten per fault family: healthy,
payment CPU, ledger pool, ledger delay, and all pair/triple combinations. Alternate
normal and shifted load/severity five times per family; shuffle collection order.
Use the existing bounded fault controls, two completed warm-up batches, reset
measurement counters without resetting faults, then four measured frames. Reject
and stop on request failure, invalid cumulative service counts, or duration >35s;
retain the failure, never silently drop/retry a case. Always reset faults in finally.

Train pinned upstream DreamerV3 from scratch: seed 51, 100,000 steps, existing
physical-v2 observation contract, eight-fault compositional actions and training-only
mechanism supervision weight 1. All weights are fitted only to the 80 training
trajectories via the empirical environment. Fixed CPU/pool/trace checks at inference;
no active check-selection claim. This is one independent initialization, not a
multi-seed confirmation or an ablation of world-model versus supervised objectives.

Fit ridge signed-label heads (penalty 1, zero threshold, training-only normalization)
on Dreamer state (mean of 16 posterior particles) and on the identical complete
observation history, flattened. Primary comparison is latent vs full-history raw
telemetry. A raw last-frame baseline and shuffled-label latent control are secondary.
All predictors use the same observed readings; labels never enter policy inputs.
Audit actual label isolation. Save fitted coefficients/checkpoint hashes before
collecting evaluation data.

Then collect 80 new incidents with scenario seed 8102, ten per family, randomized
order, all with shifted load/severity. Same clean measurement-window validation.
No fitting after evaluation starts, rerolling, exclusions, threshold selection, or
candidate replacement. Primary reliability gate: 10/10 healthy and at least 9/10
exact fault-set diagnoses in every other family. Added-value signal requires the
latent classifier to pass reliability and exceed raw full-history by >=4/80 cases;
report paired wins/losses and per-family results, without a significance claim.
Tie means no demonstrated advantage. Known-fault generalization only; no unseen
services, unknown-fault reliability, LLM integration, or remediation claim.

Only inference is paused for quota, retaining its exact template and replica count.
Application services and public UI stay running. Temporary Job has 2CPU/4Gi limits,
2h deadline and 24h TTL, no service-account token, existing runner-only network
permissions. Controls are namespace-local and leased; reset before restoration.
Archive logs, all fresh trajectories, predictions, fitted parameters, and hashes.
No serving promotion; restore inference and remove temporary Job/reader pods.
