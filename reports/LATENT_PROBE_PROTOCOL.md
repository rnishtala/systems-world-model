# Frozen latent-state probe protocol

Question: are secondary fault signals recoverable from the trained Dreamer
state, or does the diagnostic reward path fail despite retaining them?

Freeze the mechanisms-v1 seed-20 checkpoint. Extract the mean of 16 particles'
concatenated deterministic and stochastic states after CPU/pool/trace inspections.
Compare three independent binary ridge-regression probes on this latent vector
with identical probes on raw final telemetry features 2..9. Fit standardization,
intercepts and coefficients only on the original 140 calibration incidents.
Use ridge penalty 1, threshold zero, with no tuning. Add a shuffled-training-label
latent control (permutation seed 7001). Checkpoint digest must remain unchanged.

Fit and save all probes before collecting 80 fresh live incidents: seed 6400,
ten per healthy/single/pair/triple family, increased load, two warmup batches and
four recorded frames. Triple incidents remain absent from training. Extract all
model states offline after faults are reset. No labels enter model observations.
Replay original Dreamer diagnosis on the identical records as a paired reference.

Report exact sets per family and total. The exploratory reliability gate remains
10/10 healthy and at least 9/10 in each other family. Success of a latent probe
shows recoverable information and motivates a new diagnostic head. Failure of a
linear probe does not prove information is absent; probe capacity, limited data,
particle averaging and distribution shift remain alternative explanations.
Comparisons are offline diagnosis, not online end-to-end latency measurements.

Keep all changes in this repo and systems-world-model-lab. Pause only inference
while the runner uses 2.4 CPU/4 GiB; services retain 1.6 CPU, preserving the 4 CPU/
6 GiB namespace cap. Fault leases and request checks remain. Restore the prior
inference controller after completion or failure; no other namespace operations.
