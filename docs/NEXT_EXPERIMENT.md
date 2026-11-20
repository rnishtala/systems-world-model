# Direct ledger probes before diagnosis

Question: does measuring ledger behavior directly before the six-window diagnosis
reduce confusion between fixed delay and pool contention?

```mermaid
flowchart LR
  COLLECT[192 fresh ledger-first training incidents] --> TRAIN[Dreamer: seed 113, 100k steps]
  TRAIN --> FREEZE[Freeze model and readouts]
  FREEZE --> TEST[64 fresh ledger-first test incidents]
  TEST --> CAND[New candidate]
  TEST --> REF[Frozen mixed-schedule reference]
  CAND & REF --> SCORE[Exact diagnoses and pool errors]
```

Each incident: four checkout windows, two direct ledger windows, then two checkout
windows. Both low and high load are measured. Trajectories are collected in that
order; archived measurements are never rearranged to simulate new actions.

- Three new 64-incident training collections: seeds 20261121, 20261122, 20261123.
- Reference: `0fc61b145fff`, 192 training incidents, seed 113, 100,000 steps,
  six/eight-window supervision. Preserve its model and readouts unchanged.
- Keep architecture, telemetry fields, auxiliary weight and readout selection fixed.
- Candidate receives new training data as well as a different schedule. This is a
  feasibility experiment, not a causal ablation of schedule alone. The reference
  faces a schedule shift on the shared test batch; report that limitation.
- Freeze before collecting a new balanced 64-incident test batch. Compare both models
  on exactly that file. Retests retain the candidate's ledger-first schedule.

Success: existing reliability, family and shifted-load gates; at least two additional
six-window exact diagnoses and fewer false pool positives than the reference.
Also inspect missed pool faults so an apparent false-positive gain cannot hide a
recall regression. Report raw telemetry controls and eight-window accuracy. Repeat
on a fresh batch before claiming a repeatable gain.

The queue records `probe_training_seeds` and the frozen `training_source` before
launch. These are bounded operator-only fields, not arbitrary API commands. Workers
check source hashes, supervision identity and seed overlap; protocol files record the
actual training hash and comparison scope. The UI name/report identify the schedule.
The catalogue dataset field is only a lineage/preflight reference for this run;
the actual new training data is the run's `train.jsonl`.

Fresh collection adds runtime, so this protocol has a two-hour worker cap; normal
runs retain their one-hour cap. Controller recovery allows five additional minutes.
Fault leases, per-incident safety budgets, namespace isolation, cancellation and
inference restoration remain unchanged. No resources outside the lab are modified.

See [pool error analysis](POOL_ERROR_ANALYSIS.md) for the evidence behind this test.
