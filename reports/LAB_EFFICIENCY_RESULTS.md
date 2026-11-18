# Diagnostic check efficiency — fresh lab evaluation

Frozen Dreamer seed 51; readouts fitted on the preceding 80 training incidents. Stopping thresholds selected on 80 new development incidents (8201), then evaluated on 80 later incidents (8202). No model retraining or serving promotion.

| Method | Correct / 80 | Mean checks / 3 | Reliability gate |
|---|---:|---:|---|
| Dreamer state, calibrated stopping | 79 | 3.000 | Pass |
| Dreamer state, all checks | 79 | 3.000 | Pass |
| Telemetry history, calibrated stopping | 79 | 2.750 | Pass |
| Telemetry history, all checks | 79 | 3.000 | Pass |

Prespecified added-value signal: **not met**. Requires the Dreamer stopping policy to pass the reliability gate, lose at most one diagnosis against its full-check predictor, and save at least 0.5 checks against both full checking and telemetry stopping.

## Accuracy by fixed check budget

| Checks | Dreamer correct / 80 | Telemetry correct / 80 |
|---:|---:|---:|
| 0 | 13 | 20 |
| 1 | 27 | 31 |
| 2 | 58 | 59 |
| 3 | 79 | 79 |

## Calibration and paired results

- latent: threshold all-check fallback; development calibration passed. Signed ridge margins are not probabilities.
- raw: threshold 1.0; development calibration passed. Signed ridge margins are not probabilities.
- Dreamer alone correct: 0; telemetry alone correct: 0.

## Interpretation and limits

This evaluates early stopping with fixed CPU → pool → trace ordering. It does not evaluate native Dreamer planning, learned check ordering or imagined trajectories. All measurements are collected for replay; reported savings count simulated diagnostic acquisitions, not measured runtime savings. Later prefixes include later passive observations as well as newly inspected groups. The candidate also uses supervised mechanism training, so improvements cannot be attributed solely to world-model training objectives.

One model and eight known fault regimes; no claim of unseen-system or production reliability. Thresholds were frozen before evaluation and no evaluation incidents were excluded or used to tune the rules.

## Verification and operations

Job duration: 14m51s; staging and restoration add wall time. Eleven targeted tests passed. Exported checksums, data separation, freeze timestamps, observed-prefix boundaries, all scores and predictions, checkpoint identity and restoration were verified. Application services remained running; inference was restored with its original template and replica count. Temporary resources were removed.

[Frozen protocol](LAB_EFFICIENCY_PROTOCOL.md) · [Evidence verification](evidence/lab-efficiency-v1/verification.json) · [Detailed results](evidence/lab-efficiency-v1/evaluation/summary.json)

## Next useful step

Improve diagnosis from partial observations on development data before another live evaluation. At two checks Dreamer scored 58/80 versus telemetry 59/80; earlier stopping cannot recover information the diagnostic readout does not reliably use. Audit per-prefix errors and confidence on the development split, then compare a better-calibrated partial-state readout against the matched telemetry baseline. Preserve this evaluation as held out; use a new batch only after freezing the next candidate. These results do not establish whether a learned alternative check order would help.
