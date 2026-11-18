# Fresh lab diagnostic check efficiency

Frozen before collection. Isolated to systems-world-model-lab, no serving changes.
Reuse frozen Dreamer seed 51 from lab-fresh-v1, trained on its original 80 incidents.
Fit one ridge diagnostic head per prefix (zero, one, two, three checks) using only
those original 80 training incidents, for both Dreamer state and raw observation
history. Raw history contains exactly the same observed prefixes as Dreamer.
Penalty 1, signed mechanism targets, zero decision threshold. State features are
means of 16 posterior particles. Fixed order CPU, pool, trace; no native actor,
learned check ordering, or imagined-rollout benefit claim.

Collect 80 fresh development incidents, seed 8201, ten per family, all shifted.
Select an early-stopping threshold separately for each representation from
[0, .25, .5, 1, 2, 4, infinity]. Confidence is minimum absolute signed ridge score,
not a probability. At prefixes 0..2 stop when confidence >= threshold; otherwise
use all three checks. Eligible thresholds must get healthy 10/10, every other
family >=9/10, and lose at most one total correct diagnosis relative to their
full-check predictor on development. Select minimum average checks, breaking ties
by higher threshold. If none qualify, freeze the all-three-check fallback and
report that development calibration failed. Archive all candidates.

Freeze probes and stopping thresholds before collecting 80 evaluation incidents,
seed 8202, ten per family, all shifted. Never tune on the preceding 8102 evaluation
or this new evaluation. Record accuracy at each fixed budget and selected stopping
accuracy, mean checks and per-family counts. Primary success: Dreamer stopping
passes the same reliability gate, loses at most one correct case versus its own
all-three-check baseline, and saves >=0.5 mean checks versus BOTH its all-three
baseline and the telemetry stopping policy. Report paired outcomes. Small known
regime experiment, no statistical significance or production reliability claim.

Collection uses bounded leased controls, two completed warm-up batches, a counter
reset, and four measured frames. Stop and preserve evidence on invalid counts,
request failures or duration >35s; no exclusions or rerolls. Reset faults finally.
All observations are recorded for evaluation replay; stopping policies cannot see
future prefixes. Check savings are simulated diagnostic acquisitions, not measured
end-to-end wall-time savings. Each later prefix also contains a later passive frame;
this design does not separate extra checks from extra elapsed observation time.

Only inference may be paused for quota, with original template and replicas
restored. Job limit 2 CPU/4Gi, deadline 1h, no service-account token. Application
services and public UI stay running. Export predictions, trajectories, fitted
parameters, model hashes and logs. Remove temporary resources; retain checkpoints.
