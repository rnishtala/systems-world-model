# Post-warm-up measurement experiment

Frozen before collecting new incidents. The previous failed 77/80 run remains unchanged.

## Question
Does removing completed warm-up requests from telemetry resolve the healthy startup misclassification without changing the trained model or decision rule?

## Fixed protocol
- Use model seed 22, its existing 160-incident calibration, and the existing imagined-reward argmax decision. No weight, threshold, normalization or fault changes.
- After two fully awaited warm-up batches, clear only metric rows on all three services through the isolated control port. Preserve connections, fault configuration and lease expiry. Accumulate four measured batches; require exact per-service row counts n, 2n, 3n, 4n. Reject contamination rather than score it.
- Collect 80 fresh incidents with scenario seed 6701, ten per family at shifted load. All eight families, including healthy, use the same measurement rule.
- Gate: 10/10 healthy, at least 9/10 exact fault sets in each other family. Auxiliary and raw/latent probes remain secondary comparators fitted only on original calibration.
- If primary development passes, train unused seed 23 for 100,000 steps on the same calibration and evaluate 80 fresh incidents at seed 6702 under the same window rule. Otherwise skip independent confirmation.
- Training calibration still contains warm-up telemetry. This deliberate fixed-model experiment isolates measurement semantics; it does not establish equivalence to training with clean windows. Startup incidents are outside this steady-state experiment and need a separate future scenario.
- No promotion. Save all evidence and restore original service source ConfigMaps and inference replicas. All mutations stay within systems-world-model-lab. No changes to other namespaces.
