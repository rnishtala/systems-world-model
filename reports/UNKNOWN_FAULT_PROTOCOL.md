# Predictive escalation experiment — 10 November 2026

Scope: this repo and systems-world-model-lab only; unchanged 4 CPU / 6 GiB cap.
Preserve all prior checkpoints, deployments manifests and evidence.

The previous known-case confirmation is now explicitly calibration data for this
new study, not independent validation of the new guard. Use all 30 examples per
true class per model, including misdiagnoses. Fit each sensor's absolute forecast
error bound as 1.5 times the observed maximum plus 0.05 normalized units. These
are conservative empirical bounds, not calibrated probabilities or a conformal
guarantee. Coefficients are fixed before inspecting new live results.

Reuse frozen Dreamer seeds 10/11/12 and the three-inspection harness. At the final
step, select the usual highest imagined-reward diagnosis, then compare all seven
inspected sensors' forecast errors with that candidate class's learned bounds.
Escalate when any error exceeds its bound. The inference pod receives only fitted
parameters, not calibration episodes or incident truth. Trace explicit reasons.
Never treat a high imagined reward as confidence.

Fresh pilot: 15 incidents per known class, shifted load, scenario seed 5200.
Then 12 incidents each of CPU+delay (seed 5300), CPU+pool (5400), pool+delay (5500),
all shifted load. The latter two combinations were not in earlier development.
No new training, controls, labels or topology metadata enter model observations.
All models inspect the same physical incident in shuffled order, with the same
fault lease safety checks. Benchmark checklist retained as a reference.

Report per-model/per-class known accuracy, healthy false alarms, false escalation,
and escalation versus incomplete single-cause answers for each combined fault.
Gate: >=90% exact known-fault accuracy per class; >=95% healthy accuracy;
<=5% false escalation across known incidents; >=90% escalation for EACH combined
fault family. Median diagnosis under 60s and at most three inspections.
The 60 known + 36 combined physical incidents are a small fresh pilot, not a
production reliability guarantee. Do not tune bounds against these outcomes.
If the guard fails, retain the failure and restore the confirmed unguarded
controller; document why predictive contradiction alone was insufficient.
