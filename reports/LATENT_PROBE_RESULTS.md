# Frozen-state probe result

**Dreamer retains some secondary-fault information, but the replacement probe
is not reliable enough to deploy.** No Dreamer weights changed.

| Method | Exact diagnoses on 80 fresh incidents | Gate |
|---|---:|---|
| Existing Dreamer reward path | 50/80 (62.5%) | Fail |
| Raw telemetry linear probes | 76/80 (95%) | Fail |
| Frozen latent-state linear probes | 63/80 (78.75%) | Fail |
| Shuffled-label latent control | 15/80 (18.75%) | Fail |

| Family | Dreamer | Raw | Latent | Shuffled |
|---|---:|---:|---:|---:|
| Healthy | 10/10 | 10/10 | 10/10 | 0/10 |
| CPU | 10/10 | 10/10 | 5/10 | 4/10 |
| Pool | 10/10 | 10/10 | 10/10 | 2/10 |
| Delay | 10/10 | 10/10 | 10/10 | 2/10 |
| CPU + delay | 0/10 | 10/10 | 8/10 | 1/10 |
| CPU + pool | 0/10 | 9/10 | 6/10 | 3/10 |
| Pool + delay | 10/10 | 10/10 | 8/10 | 1/10 |
| Triple, absent from training | 0/10 | 7/10 | 6/10 | 2/10 |

## What this tells us

The latent probe recovered 20/30 CPU-containing combinations that the original
reward path never identified completely. Thus secondary-fault information is
not entirely absent from the state. However, the probe introduced errors on
CPU-only and pool-plus-delay incidents; it does not isolate the reward readout
as the sole problem or establish that the representation is fully sufficient.

A training-fit diagnostic found **140/140 fitted targets correct for every
probe, including shuffled labels**. With 640 latent dimensions and only 140
training incidents, the probe can memorize noise. Training accuracy is therefore
not evidence of generalization. The shuffled control's fresh-test score is a
negative control from one permutation, not a formal significance test.

Next: fit a regularized multi-label diagnostic head on frozen states, select
regularization and capacity using training-only cross-validation, then evaluate
on a new held-out batch. Keep this test seed (6400) out of that selection.
Do not spend another full Dreamer training cycle before testing this cheaper fix.

## Method and limits

The original 140 calibration incidents fitted all probes. We froze the probes
before collecting seed-6400 incidents: ten per family, increased load, two
warmup batches and four recorded frames. Only inference paused; live application
services stayed running. The checkpoint's before/after SHA-256 matched.

Latent features are the mean concatenated deterministic/stochastic state across
16 particles after the fixed CPU/pool/trace inspections. Each mechanism has a
binary ridge probe; penalty 1 and threshold 0 were fixed without tuning. The
raw probe uses the eight prespecified load/inspection features from the earlier
baseline. The latent model also sees observation history, front-end latency/error
signals and masks, and has greater dimensionality. This is a decodability study,
not a capacity- and information-matched proof of superiority over raw features.
A linear probe failure cannot prove information is absent or irrecoverable.

All four methods are scored offline on the same fresh records. No online latency
claim is made. No model passed 10/10 healthy plus at least 9/10 in every other
family. One model and ten incidents per family do not establish reliability.

## Validation and artifacts

- 33 regression tests passed, including label-invariant observations, fixed
  training-only normalization and finite constant-feature behavior.
- Fresh collection completed without invalid requests or expired fault windows.
- All work stayed in this repository and `systems-world-model-lab`, under the
  unchanged 4 CPU/6 GiB quota. The prior inference deployment was restored.
- [Protocol](LATENT_PROBE_PROTOCOL.md), [evidence](evidence/latent-probe), and
  [research note](../output/pdf/dreamerv3-latent-probes.pdf).

The evidence includes the exact training/test feature arrays, fitted coefficients,
fresh raw trajectories, paired predictions, hashes, tests and workflow logs.
