# Auxiliary mechanism supervision: development results

The auxiliary head scored **71/80**, the same model’s reward readout **77/80**, and telemetry alone **77/80**. Auxiliary development gate: **FAIL**. No candidate was promoted.

| Family | Auxiliary | Reward | Telemetry |
|---|---:|---:|---:|
| Healthy | 10/10 | 10/10 | 10/10 |
| CPU | 10/10 | 10/10 | 10/10 |
| Pool | 10/10 | 10/10 | 10/10 |
| Delay | 10/10 | 10/10 | 10/10 |
| CPU + delay | 10/10 | 10/10 | 10/10 |
| CPU + pool | 10/10 | 10/10 | 10/10 |
| Pool + delay | 10/10 | 10/10 | 10/10 |
| Triple | 1/10 | 7/10 | 7/10 |

Training seed 21: 100,000 total steps; resumed-process duration 6.33 minutes (earlier training and downtime excluded). Original 140 calibration incidents only. Fresh evaluation seed 6501: 80 physical incidents, ten per family, increased load/severity; triple absent from training.

The new loss is independent binary cross-entropy for CPU, pool and delay (weight 1), trained through the shared Dreamer posterior. Labels never enter encoder, decoder or inference API inputs. The actual upstream smoke checkpoint reloaded successfully; opposing label placeholders produced identical inference outputs. Evaluation also audited the full checkpoint, and its digest was unchanged afterward.

Exploratory frozen latent probe: 72/80; shuffled-label latent control: 15/80. All probes were frozen before collection. The telemetry baseline is not capacity- or information-matched to the recurrent model.

All three primary methods scored 70/70 on healthy, single and pair families. All errors were on the unseen triple. The auxiliary head omitted delay in nine triples; the reward readout omitted delay twice and pool once; telemetry omitted pool three times. Reward and telemetry have equal totals but disagree on four incidents. This is a supervised diagnostic head on a world model, not evidence that imagined planning improves triage. Fixed inspections and offline scoring of real trajectories do not demonstrate online resolution or learned interventions. One seed and small topology cannot establish reliability; older model scores used different batches and are not controlled ablations.

Next: separate practical known-regime coverage from zero-shot composition. Adding triple-fault calibration followed by fresh held-out incidents could test known-regime reliability; it would no longer test an unseen triple. Unseen-combination generalization needs a separate controlled study. A matched no-supervision ablation and independent seeds remain pending. Do not tune thresholds against this batch.

Validation: 26 model/environment tests and four monitor regression tests passed. All 80 fresh records and reported scores were independently checked. The interrupted seed-6500 run retains 23 completed records without an accuracy claim. All five lab deployments were restored to the existing baseline. No other namespaces or repositories were changed.

Evidence: `reports/evidence/auxiliary/`. Protocol: `reports/AUXILIARY_SUPERVISION_PROTOCOL.md`. Research PDF: `output/pdf/dreamerv3-auxiliary-supervision.pdf`.
