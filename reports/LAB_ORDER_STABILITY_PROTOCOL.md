# Development-only order and confidence stability audit

2026-11-18. Frozen before executing this audit.

Use only seed 8201 development incidents, lab-order-v1 development features,
and the already frozen final readouts/policy. Never load confirmation incidents
or features. No cluster operations or Dreamer training.

1. Reproduce final policy selection with five folds and seed 8400; verify selected
   order, threshold, and readout configurations against the frozen manifest.
2. Compare its out-of-fold stopping decisions with the frozen all-development-fit
   readouts applied to the same development cases. Report score margins per prefix.
   The latter is an in-sample diagnostic, NOT a generalization estimate.
3. Repeat the unchanged selection procedure with prespecified seeds 8500–8509.
   Report every selected order/threshold and selection frequency. Do not pick a
   winning seed, add candidates, tune thresholds, or launch fresh confirmation.
4. Hash inputs before/after and preserve results. Interpret score changes as a
   possible calibration-transfer problem, not proof of the fresh-test failure's cause.

If confidence fails to transfer, a subsequent experiment should fit readouts first,
calibrate on separate development cases, and freeze without refitting afterwards.
