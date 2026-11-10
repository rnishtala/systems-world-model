# Complete fault-regime coverage: results

Twenty newly collected triple-fault incidents augment the original 140 calibration records. No previous held-out incidents enter training. This evaluates new incidents across known regimes, not unseen fault combinations.

## Development: model seed 22

Primary reward readout: **77/80**; gate **FAIL**. Auxiliary: 78/80. Telemetry: 78/80. Training: 12.37 minutes, 100,000 steps.

| Family | Reward | Auxiliary | Telemetry |
|---|---:|---:|---:|
| Healthy | 9/10 | 9/10 | 9/10 |
| CPU | 10/10 | 10/10 | 10/10 |
| Pool | 10/10 | 10/10 | 10/10 |
| Delay | 10/10 | 10/10 | 10/10 |
| CPU + delay | 10/10 | 10/10 | 10/10 |
| CPU + pool | 9/10 | 10/10 | 10/10 |
| Pool + delay | 10/10 | 10/10 | 10/10 |
| Triple | 9/10 | 9/10 | 9/10 |

Independent confirmation was skipped because the prespecified primary development gate failed. Comparators do not override that gate.

Post-hoc error analysis: all three primary methods called the first healthy incident delay. Its ledger cumulative p95 was about 440 ms, while its measured batches completed in 60–70 ms. Metric counts include the two warm-up batches, and service snapshots retain samples until the next control reset. Thus elevated warm-up latency persisted into the measured observations; the specific startup cause is not proven. This row remains in the score. The shared triple miss omitted pool, and the reward readout additionally missed pool in one CPU+pool case.

Next: define and test a consistent measurement window that excludes warm-up samples without resetting fault state or pool connections, retain startup transients as a separately evaluated scenario, and then run a fresh validation. Investigate weak pool-pressure observations rather than relabeling or dropping failed cases. Independent confirmation remains pending.

The primary readout and gates were fixed before data collection. Passing a small batch would support the bounded demonstration, not production reliability or statistical proof of 90% per-family accuracy. Equal or worse performance than telemetry alone does not establish added value from imagined planning. A matched no-supervision ablation, broader topologies, unseen causes, LLM integration and autonomous repair remain outside this result.

All label-isolation audits passed; checkpoint digests were unchanged during scoring. The previous deployment was restored, and no candidate was promoted. Other namespaces and repositories were untouched.

Protocol: `reports/COMPLETE_COVERAGE_PROTOCOL.md`. Evidence: `reports/evidence/complete-coverage/`. PDF: `output/pdf/dreamerv3-complete-coverage.pdf`.
