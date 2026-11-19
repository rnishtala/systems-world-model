# Frozen Dreamer stopping development results

Optimistic development screen: world-model weights trained on these 64 incidents; nested readout/stopping selection only.

| Method | Exact diagnoses | Mean windows | Mean ingress requests | Full-eight accuracy | Calibration failures |
|---|---:|---:|---:|---:|---:|
| latent | 64/64 | 7.48 | 44.95 | 64/64 | 0 |
| history | 34/64 | 8.00 | 48.00 | 34/64 | 4 |
| summary_history | 51/64 | 8.00 | 48.00 | 51/64 | 4 |

Preregistered development gate: **failed**.

Costs are retrospective prefix estimates on an unchanged measured schedule. Warmup is excluded. No live early-stopping latency or alternative-action planning is demonstrated.

The model was frozen throughout. No new faults were injected and no model was promoted. Only training incidents were used; previous test9202 was not used for selection.

Nested readout/stopping selection protects outer-fold labels, but the world model previously trained on all64 trajectories. Treat these as optimistic development results, not fresh generalization evidence.

Next: do not spend a fresh confirmation run on this stopping rule. Investigate whether earlier states encode the fault mechanisms, and improve the training objective or probe design before another bounded experiment.

Evidence: reports/evidence/probe-stopping-v1. Protocol: reports/PROBE_STOPPING_PROTOCOL.md.

## Interpretation and next candidate

The stopping rule saves 0.516 of eight windows (6.4%) and 3.047 of48 ingress requests (6.3%). It fails the required one-window saving, despite64/64 development diagnoses.

A post-hoc candidate is the fixed six-window Dreamer readout:64/64 in these same outer folds, with36 instead of48 measured ingress requests. A fixed six-window budget would save25% of measured requests/windows, excluding warmup. This was discovered after the screen and is not a validated saving; freeze it separately and collect fresh confirmation before claiming it.

Telemetry controls failed all four inner calibration gates and therefore fell back to eight windows. Their weak grouped-CV results do not establish Dreamer superiority: world-model weights saw all development trajectories, and leaving out repeats shifts load/order schedules. The earlier fresh test had telemetry and Dreamer tied32/32. Preserve that result; do not reinterpret this optimistic development screen as a new accuracy win.
