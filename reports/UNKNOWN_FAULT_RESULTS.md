# Predictive escalation pilot results

The fixed pilot did not pass all gates. The guard is retained as an experimental option; the confirmed `inspect-first` controller is restored. No weights were retrained.

## What improved

All models escalated all 12 CPU+delay and all 12 CPU+pool incidents. Known-case false escalation was zero out of 60 per model. The guard uses disagreement between forecasts and observations, not imagined reward as a confidence probability.

| Seed | Healthy | CPU | Pool | Delay | Pool+delay escalation |
|---|---:|---:|---:|---:|---:|
| 10 | 100.0% | 100.0% | 100.0% | 86.7% | 100.0% |
| 11 | 100.0% | 100.0% | 100.0% | 93.3% | 83.3% |
| 12 | 100.0% | 100.0% | 100.0% | 86.7% | 83.3% |

## Why the candidate was not promoted

Seeds 10 and 12 misclassified 2/15 known ledger-delay cases as pool contention; seed 11 missed 1/15. All were inside the selected class’s residual bounds, so the guard did not reject them. For combined pool+delay, seeds 11 and 12 returned only pool contention in the last two incidents (10/12 escalations). Seed 10 escalated all 12. These failures breach the fixed per-class criteria.

The underlying model candidate and guarded decision used the same observations. There were no known-case escalations, so the fresh diagnosis errors were already present before applying the guard. Restoring the older controller preserves prior behavior; it does not fix those newly observed errors.

The earlier 99.2–100% confirmation remains valid for that dataset, but this fresh stream does not reproduce its per-class reliability. We do not tune the guard on these outcomes or replace the old evidence.

## Method and verification

The former 120-incident confirmation was explicitly repurposed as calibration for this new study. For each seed, true class and sensor, bounds were frozen at 1.5 times its maximum absolute forecast error plus 0.05 normalized units. Runtime selects bounds using the candidate diagnosis, then escalates on any exceeded sensor bound. Raw calibration episodes and incident truth are not mounted in inference. This is an empirical harness, not a statistical confidence guarantee.

Fresh validation used 60 known incidents and 36 combined incidents (12 per family). Three models share each incident: 96 physical incidents, 288 model evaluations, plus 96 checklist evaluations. The common incident stream and shared calibration make outcomes dependent.

All 19 tests passed, including missing/nonfinite evidence rejection and preservation of both pool and delay controls. All experiment jobs ended, bounded faults reset, and the temporary evidence reader was removed. Source bundle v11 and fitted-parameter hashes are recorded in `evidence/guard/provenance.json`.

## Next experiment

Separate CPU, pool and delay mechanisms in the learned representation and diagnostic targets, rather than requiring one mutually exclusive cause. Collect broader delay/pool trajectories across load and latency overlap. Evaluate combinations and abstention on new held-out families/seeds. Keep the existing threshold guard as a measured baseline; do not infer that more restrictive bounds alone solve unfamiliar-fault recognition.

All work remains in this repository and `systems-world-model-lab`, under the unchanged 4 CPU / 6 GiB cap. No AutoSRE/BearBank namespace or repository was changed.

Research PDF: `../output/pdf/dreamerv3-predictive-escalation.pdf`. Full scores and raw traces: `evidence/guard/`.
