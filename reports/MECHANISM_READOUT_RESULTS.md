# Targeted diagnostic readout experiment

2026-11-18. [Protocol](MECHANISM_READOUT_PROTOCOL.md) · [Evidence](evidence/mechanism-readout-v1/summary.json).

Selecting a different representation and regularization for each fault mechanism did not improve the existing diagnostic readout. The prespecified development gate failed; no live collection, retraining or promotion followed.

| Readout | Exact diagnoses | Combined faults (64) | CPU / pool / delay bit errors |
|---|---:|---:|---:|
| Mechanism-specific, world-model features allowed (primary) | 112/128 | 55/64 | 1 / 8 / 10 |
| Mechanism-specific, telemetry only | 104/128 | 46/64 | 1 / 16 / 13 |
| Existing shared-config latent readout | 119/128 | 59/64 | 0 / 8 / 4 |
| Existing load-response readout | 104/128 | 47/64 | 1 / 18 / 12 |

The primary improved over the matched telemetry selector by eight incidents but regressed seven against the existing latent readout. It also missed the 116/128 overall threshold and the minimum for pool-only (10/16) and CPU-plus-delay (11/16). This result does not justify a new confirmation run for this readout.

These are nested workload-group readout development scores, not fresh model evaluations. Model 81 trained on all 128 incidents. The earlier independent 64-incident result remains the generalization evidence: latent 53/64, load response 58/64, forecast-feature control 59/64. No weights improved in this experiment.

## What the existing fresh errors reveal

A separate descriptive audit of the completed confirmation found zero CPU-presence errors in all three readouts. Latent made nine pool-bit errors and five delay-bit errors across eleven wrong incidents (some incidents have two wrong bits). Its errors were:

- CPU + pool mistaken for CPU + delay: three.
- CPU + delay mistaken for all three: four.
- Pool mistaken for healthy: one.
- Pool mistaken for pool + delay: two.
- All three mistaken for CPU + delay: one.

Thus the immediate difficulty is identifying pool contention versus delay, including when CPU load is present. The audit does not prove why the signals overlap. More classifier flexibility on the same observations did not resolve it.

## Next discriminating experiment

Test a targeted, measured intervention before further Dreamer training: compare checkout-driven load with direct ledger requests at the same two declared concurrency levels. Payment CPU work occurs before downstream requests and may change their arrival pattern; a direct ledger probe removes that upstream influence. Pool contention should then show a different concurrency response from an injected delay. This is a hypothesis, not an established finding.

First use a balanced pilot across all eight families, preserving fault leases, successful-request checks, namespace limits and reset-on-exit. Compare the extra probe against an equal-request-budget repeat of the current checkout probe, so any benefit cannot be attributed simply to more observations. Fix the comparison and success criteria before collection. Use trace latencies only, excluding injected settings, direct pool counters and labels from diagnostic inputs.

Only if this separates the ambiguous cases on fresh evidence should we collect corresponding training trajectories and extend Dreamer's action/observation contract to represent probe target and concurrency. The existing workload-only checkpoint cannot validly simulate an untrained ledger-target intervention. Both world-model and telemetry controls must receive the same new observations, followed by a separate frozen test. This experiment was not launched in this readout-only step.

## Verification

The verification script reproduced every inner candidate score, selection, outer prediction and metric; checked disjoint nested folds and source hashes; and matched both original baseline prediction arrays exactly to the previous development experiment. The confirmation audit ran separately after development and did not feed selection. No cluster resources or serving configuration were touched.
