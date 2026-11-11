# Frozen classifier integration and live API validation

Prespecified before candidate API replay or fresh collection. Model seed 23 and its independently confirmed classifier remain unchanged. No training, fitting, threshold adjustment or candidate selection is permitted in this experiment.

1. Package the seed-23 model configuration, checkpoint, training metadata and four frozen latent classifier arrays only. Exclude calibration, held-out trajectories and replay. Bind the classifier to checkpoint and artifact hashes and fail closed on mismatches.
2. Temporarily run controller `inspect-latent` through the existing inference-only HTTP API in systems-world-model-lab. Keep its agent network role, read-only model mounts and no service-account credentials. The collector alone has fault controls.
3. Replay all 80 preserved seed-6801 observation sequences through the HTTP API. Require the same inspection actions (1,2,3), the same final diagnosis as the saved primary predictions, and the expected classifier artifact hash. This is an integration check, not new accuracy evidence. The server starts normally; do not feed it calibration data to align random state.
4. Require invalid requests containing fault labels or correctness rewards to be rejected. Only if all replay/contract checks pass, collect 80 fresh live incidents, ten per family, scenario seed 6901, shifted load/severity. Warm up twice, clear metrics only, then interleave four measured frames with real HTTP inference calls. Never send fault labels or control parameters to the API.
5. Check every service's cumulative counts n,2n,3n,4n, all requests successful, total injection-to-diagnosis duration <=35 seconds. Stop on invalid incidents; do not omit/relabel them. Evaluate exact fault sets; healthy requires 10/10 and each other family >=9/10. Preserve every case and response. No reroll after a failed gate.
6. Restore original application sources and original inference deployment after the experiment, whether pass or fail. No automatic persistent promotion. All work and artifacts stay in this repository and namespace.

This verifies integration of the confirmed fixed-inspection classifier, not learned investigation planning, startup handling, unseen services, LLM reasoning or autonomous remediation. Historical test observations used for replay are never fitted or counted as fresh test successes.
