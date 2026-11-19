# Dreamer trained on expanded incident coverage

Trained a fresh upstream Dreamer checkpoint (seed 101, 100,000 steps) in 17.5 minutes on 128 designated training incidents. World-model weights changed. The original 64 and expanded-range 64 training incidents were reused without additional collection; all completed test datasets were excluded.

Frozen six/eight-window diagnostic readouts were fitted using training-only selection. A fresh 64-incident test (seed 9503) evaluated the new checkpoint, the original Dreamer91 readout, its expanded-data refit, and matched telemetry controls on identical incidents.

| Method | Correct / 64 | False pool diagnoses | CPU / pool / delay bit errors |
|---|---:|---:|---|
| New Dreamer101, six windows | 61 | 2 | [0, 3, 0] |
| New Dreamer101, eight windows | 64 | 0 | [0, 0, 0] |
| Telemetry, six windows | 57 | 5 | [0, 7, 0] |
| Telemetry, eight windows | 62 | 2 | [0, 2, 0] |
| Dreamer91, original readout | 61 | 3 | [0, 3, 0] |
| Dreamer91, expanded-data readout | 60 | 3 | [0, 4, 0] |

| Regime | New latent6 | Old + expanded readout | Old + original readout | Telemetry6 |
|---|---:|---:|---:|---:|
| Original | 31/32 | 31/32 | 32/32 | 31/32 |
| Higher load/severity | 30/32 | 29/32 | 29/32 | 26/32 |

Preregistered engineering gate: **failed**.
Reliability: True; per-regime requirements: False; improvement over expanded-data Dreamer91 readout: False; matched controls: False.

| Paired comparator | New-only correct | Comparator-only correct | Unadjusted exact p |
|---|---:|---:|---:|
| New Dreamer101, eight windows | 0 | 3 | 0.2500 |
| Telemetry, six windows | 7 | 3 | 0.3438 |
| Telemetry, eight windows | 2 | 3 | 1.0000 |
| Dreamer91, original readout | 1 | 1 | 1.0000 |
| Dreamer91, expanded-data readout | 2 | 1 | 1.0000 |

## Interpretation

The new six-window model changed exact diagnoses by +1/64 relative to the expanded-data readout on the old world model. False pool predictions changed from 3 to 2. Use this same-test comparison rather than comparing scores from different incident batches.

This is one training seed and one fresh incident batch on the same service architecture and known fault families. Dataset size and model initialization both differ from the older world model, so this does not isolate a causal data-volume effect. Readout selection remains supervised. Paired p-values are exploratory and unadjusted across comparisons; engineering thresholds do not establish statistical superiority. No held-out architecture, unseen fault family, forecast quality improvement or end-to-end resolution claim is measured here.

All eight windows were collected and six-window diagnosis was evaluated from the prefix. Actual early stopping and new latency savings were not measured. Labels are excluded from encoder/decoder inputs and verified by a runtime label-invariance audit. The action contract still represents a fixed measured probe schedule, not unrestricted counterfactual planning.

Evidence: reports/evidence/probe-retrain-v1. Protocol: reports/PROBE_RETRAIN_PROTOCOL.md. Reproduction: scripts/verify_probe_retrain.py. Checkpoint: /results/probe-retrain-v1/models/final-101 on the lab results PVC. Serving model remains unchanged; temporary resources are cleaned up and inference is restored.

## Decision and next experiment

The six-window improvement gate failed: only one extra correct diagnosis over the expanded-data baseline, two false pool diagnoses remain, shifted CPU+delay is only 2/4 correct, and six windows trail eight by three cases. Do not promote this checkpoint or claim reliable early stopping.

All three six-window mistakes concern the pool mechanism. Eight windows correct all three on this batch, while eight-window telemetry scores 62/64. This suggests additional observations remain useful, but a perfect 64-case result is not a general reliability guarantee. The old model was compared at six windows only, so this run does not establish a full-budget improvement over the old model.

Next, test a training-calibrated selective stopping policy: stop at six windows only when training-only confidence/consistency checks justify it; otherwise acquire the remaining two windows. Freeze the policy before a new matched actual-budget test against always-eight Dreamer and telemetry baselines. Require reliability per fault family/regime and meaningful actual request/time savings, not just pooled accuracy. Do not choose a confidence threshold using this completed test. If training validation cannot identify a useful stopping policy, retain all eight observations rather than launch another unchanged training run.
