# Direct-ledger probe pilot

2026-11-18. Freeze this protocol before collecting new incidents.

Hypothesis: upstream CPU work alters downstream arrival patterns, obscuring pool contention versus delay. A direct ledger probe may distinguish them better than repeating the checkout probe. This is an information-acquisition experiment, not Dreamer training or evaluation.

Collect 32 development incidents (seed 9101) and 32 fresh test incidents (seed 9102), each with four independently configured incidents per family. Keep existing bounded severity ranges, 45-second fault leases, reset in finally, and a 35-second incident limit. No rerolls. Four repeats per family balance extra-probe order and low/high phase. Both splits use loads 3 and 9. Randomize incident order independently.

For each incident, warm checkout and ledger with two high-load batches each; discard these measurements. Then collect four checkout base windows (alternating load), two additional checkout windows and two direct ledger windows. Counterbalance which extra target runs first. Reset metrics before each batch; require all requests successful, correct per-service counts, and completion inside the lease. Store only trace latency, error and count measurements. Both diagnostic arms use the same base observations and 12 extra ingress requests (3+9); ledger/database request counts also match. Upstream service work differs intentionally. Wall-clock cost is recorded, not assumed equal. Combined collection shares state across arms; counterbalancing limits but does not eliminate order effects.

Feature contract: base load-response features (14 existing dimensions) plus low/high log ledger latency and its concurrency slope (3 dimensions), from either extra checkout or direct ledger probes. Both arms receive 17 dimensions; base-only is a secondary control. No direct CPU, pool-wait or database-time counters, fault settings, labels, seeds or repeat IDs enter features.

Select each arm's shared three-output readout among the existing eight candidates using leave-one-repeat-out CV on development incidents only; normalize within training folds. Fit all development incidents, save immutable readouts before starting test collection. Same fitting and selection budget for both arms. Never tune on test labels. Primary outcome: exact pool-and-delay diagnosis (CPU bit ignored) on 32 test incidents. Screening gate: direct ledger >=28/32 and at least four more correct than extra checkout; healthy pool/delay 4/4 and every family >=3/4. Also report full three-mechanism accuracy, per-bit errors, paired wins/losses and exact McNemar p, cost and results by extra-probe order. This small pilot cannot establish production reliability.

If the pilot fails, stop without widening thresholds or retraining. If it passes, the next step is collecting probe-target-aware trajectories and updating Dreamer's action contract, followed by a new frozen comparison with equally informed telemetry controls. The existing checkpoint cannot simulate this new intervention.

All Kubernetes operations remain in systems-world-model-lab. Temporarily pause inference to respect quota, restore its exact template and replica count, remove temporary job/configmap resources, and verify lab health. No PVC or model checkpoint access is needed.
