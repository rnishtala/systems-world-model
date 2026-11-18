# Log-scaled trace diagnosis with separate workload-regime validation

2026-11-18. Freeze before collection/training. Only systems-world-model-lab.

Keep the Dreamer architecture and 100,000-step training budget. Train seed 71
from scratch using new measurements only; do not load any earlier test dataset.
Replace the old capped latency view with log1p(seconds / 0.1), directly from measured
checkout latency, payment self-time, ledger latency and payment downstream time.
Expose checkout error rate and observed batch request count / 12 as before. No hard
cap, CPU/pool/DB counters, controls, fault labels, regime IDs or seed enter model inputs.
The scale is fixed here, not fitted to validation or test. Keep 14 input dimensions.

Collect training seed 8701: 128 incidents, four regimes [2,4,2,4], [4,6,4,6],
[6,8,6,8], [8,10,8,10]. Each family has four incidents per regime, two in each
forward/reversed phase. All eight known fault families use the usual unshifted
fault-severity ranges. Train the world model only on this split.

After training, collect validation seed 8702: 64 incidents, regimes [3,7,3,7]
and [5,9,5,9], four incidents per family/regime with balanced phases. These regimes
and their distinct load levels are absent from model/readout fitting. Fit each of
the existing eight linear/RBF readout candidates on TRAIN ONLY. Select by validation
exact-diagnosis accuracy, ties in existing candidate order. Keep the selected fitted
readout as-is: no refit on validation. This validates the complete frozen learned
representation, unlike random readout folds on representation-training incidents.
Selection-set performance is still optimistic; final testing remains necessary.

Compare Dreamer final state, full telemetry history, latest telemetry snapshot,
auxiliary Dreamer head, and shuffled-label latent control. Each sees the same four
windows. Labels only supervise training. Check actual upstream label invariance,
training/data/checkpoint/readout hashes, and previous wait-action alignment.

Before any final-test collection, freeze choices and apply this screening gate:
Dreamer >=58/64 exact; healthy 8/8; each other family >=6/8; and Dreamer at most
two correct diagnoses behind the stronger history/latest baseline. If it fails,
stop with a validation result and preserve the final test as uncollected. This
screen is less strict than final reliability; it avoids a costly confirmation of
an already weak policy. No further tuning or training within this experiment.

If screening passes, collect seed 8703: 80 fresh incidents, ten per family,
balanced [3,11,6,9] and reverse. This is an unseen mixed workload sequence, including
load 11 unseen in training/validation, not entirely unseen loads. Final gates:
healthy 10/10, >=9/10 each other family, >=4 more correct than the stronger telemetry
baseline. Report paired wins/losses, exact paired test and accuracy intervals.
No automatic promotion. A single seed/small known-fault lab is exploratory evidence.

Every incident: reset controls, two completed warmups, reset metrics separately
before each measured batch. Preserve and fail on request/count errors or >35 seconds;
no exclusions/rerolls. Existing faults expire after 45 seconds; reset in finally.
Only temporary namespaced jobs, current quota and results PVC. Restore the existing
inference template/replicas, retain UI availability, remove temporary pods/jobs.
Evidence is exported read-only with bounded chunks; no repeat experiments for export.
