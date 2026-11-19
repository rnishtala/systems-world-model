# Fresh six-observation confirmation

Frozen Dreamer model91 and readouts selected only on64 training incidents;32 new incidents, seed9302. Same fault families/topology and load distribution as training. No model training or serving promotion.

| Method | Exact diagnoses | Observations |
|---|---:|---:|
| latent6 | 32/32 | 6 |
| latent8 | 32/32 | 8 |
| history6 | 30/32 | 6 |
| history8 | 32/32 | 8 |
| summary_history6 | 29/32 | 6 |
| summary_history8 | 32/32 | 8 |
| response8 | 32/32 | 8 |

Prespecified feasibility gate: **passed**.

| Compared with | Six-window Dreamer only correct | Control only correct | Paired p |
|---|---:|---:|---:|
| latent8 | 0 | 0 | 1.0000 |
| history6 | 2 | 0 | 0.5000 |
| history8 | 0 | 0 | 1.0000 |
| summary_history6 | 3 | 0 | 0.2500 |
| summary_history8 | 0 | 0 | 1.0000 |
| response8 | 0 | 0 | 1.0000 |

Six measured windows imply36 ingress requests versus48 for eight:25% less measured evidence. Including36 warmup requests, the reduction is14.3%. These are shadow estimates: this experiment collected all eight windows, so actual latency savings have not been demonstrated.

The six-window candidate was selected from an earlier optimistic development screen. All parameters were frozen before this new collection; test labels were used only for scoring. The small32-incident sample supports an engineering feasibility assessment, not statistical noninferiority, production reliability, unseen-service generalization, or adaptive probe planning.

The primary gate permits one fewer correct diagnosis than the strongest full-eight control. A gate pass therefore does not establish superior accuracy or a unique world-model benefit; the equally budgeted telemetry results above must be considered.

Collection wall time within incidents: 277.5s total; max 10.99s. Includes warmup and eight-window measurement, excludes model initialization/extraction and infrastructure lifecycle.

Protocol: reports/PROBE_SIX_PROTOCOL.md. Evidence: reports/evidence/probe-six-v1.

## Interpretation and next step

The six-window Dreamer readout matched full-eight accuracy on this fresh sample. Compared with equally budgeted telemetry, its two/three additional correct diagnoses yield paired p-values0.50/0.25, so this is an encouraging signal rather than established superiority.

Next: keep these readouts frozen and run a larger, balanced comparison that actually stops collection at six windows. Measure warmup-inclusive requests and latency, and include harder severity/load shifts. Do not promote the model on this32-incident result alone.
