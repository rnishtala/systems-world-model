# Development order stability and training status

2026-11-18. Protocol: [development-only audit](LAB_ORDER_STABILITY_PROTOCOL.md).
Evidence: [summary](evidence/lab-order-stability-v1/summary.json).

## Where training stands

The latest experimental Dreamer checkpoint (seed 51) completed 100,000 training
steps from scratch in 1,024 seconds (17m04s). Training used an empirical simulator
calibrated from 80 measured lab incidents, with training-only mechanism supervision.
Recent experiments freeze Dreamer and fit diagnostic readouts / inspection policies;
they do not update Dreamer weights. The existing serving model remains model 23.

Known-fault diagnosis is demonstrated: fresh confirmation scored 80/80 for Dreamer
and 79/80 for telemetry. Efficient investigation is not demonstrated: Dreamer used
2.9625 checks versus telemetry's 1.5375. These are simulated acquisition counts,
not measured time savings. Unseen-system generalization, useful imagined action
selection, and LLM-driven end-to-end resolution remain unproven.

## What this audit found

| Method | Selected out-of-fold checks | Final-fit in-sample checks |
|---|---:|---:|
| Dreamer, pool → CPU → trace | 2.95 | 2.95 |
| Telemetry, trace → pool → CPU | 1.6875 | 1.50 |

Final seed-8400 order, threshold and readout choices reproduced exactly. Input
hashes match the frozen manifest and were unchanged after execution. No confirmation
observations or features were read. In-sample figures are diagnostic, not validation;
selected out-of-fold figures also reuse data for selection, so are not independent
performance estimates.

The conjecture that refitting caused a collapse in Dreamer's stopping behavior is
not supported here: both score sets stop after 2.95 checks. The final policy was
already inefficient before fresh confirmation. Only two development cases crossed
its threshold after the first check; none crossed it after the second.

Across the ten prespecified split seeds, Dreamer selected five different orders:
CPU/pool/trace once, pool/trace/CPU twice, CPU/trace/pool once, pool/CPU/trace five
times, and trace/CPU/pool once. Telemetry split evenly between trace/pool/CPU and
pool/trace/CPU. This measures sensitivity on the same small development set,
not ten independent experiments. No seed was chosen as a new winner.

## Corrected launch gate and next experiment

The nested development comparison passed its half-check advantage gate, but the
final full-development selection did not retain that advantage. The launch gate
only required calibrated Dreamer stopping with fewer than three checks. That was
too weak. The final-policy fitter now requires the original reliability criteria
and at least half a check saved against BOTH full inspection and the final telemetry
policy. The historical final policy fails this gate. Historical results and frozen
artifacts remain unchanged. Nine focused tests passed, including this regression.

Next: develop a policy with readout fitting and stopping calibration separated,
freeze it without refitting after calibration, and evaluate selection stability
before any fresh live run. Use new development incidents if existing data are too
small; keep every prior confirmation set out of fitting. Further Dreamer training
should target an identified representation/dynamics deficiency rather than simply
increase training steps. No new training or cluster jobs were started by this audit.
