# Direct-ledger acquisition pilot

2026-11-18. [Protocol](LEDGER_PROBE_PROTOCOL.md) · [Evidence](evidence/ledger-probe-v1/summary.json).

The prespecified screening gate passed. This tests the value of an extra observation, not Dreamer performance. No Dreamer weights were trained or checkpoint promoted.

Thirty-two development incidents selected and fitted the classifiers. Their parameters were frozen before 32 fresh test incidents. Each split contains four cases per fault family. Both extra probes used 12 ingress requests and 12 ledger/database requests; the base checkout observations were shared.

| Test readout | Pool + delay exact (primary) | All three mechanisms exact | CPU / pool / delay bit errors |
|---|---:|---:|---:|
| Base checkout only | 29/32 | 29/32 | 0 / 3 / 1 |
| Base + repeated checkout | 28/32 | 28/32 | 0 / 4 / 0 |
| Base + direct ledger | 32/32 | 32/32 | 0 / 0 / 0 |

| Family | Repeated checkout: pool/delay | Direct ledger: pool/delay |
|---|---:|---:|
| Healthy | 4/4 | 4/4 |
| CPU | 4/4 | 4/4 |
| Pool | 4/4 | 4/4 |
| Delay | 4/4 | 4/4 |
| CPU + delay | 3/4 | 4/4 |
| CPU + pool | 2/4 | 4/4 |
| Pool + delay | 4/4 | 4/4 |
| All three | 3/4 | 4/4 |

On the primary outcome, direct ledger was uniquely correct on 4 cases; repeated checkout was uniquely correct on 0. Two-sided exact McNemar p=0.1250. This small sample does not establish production reliability.

The screening gate requires direct ledger >=28/32, a gain of at least four over repeated checkout, healthy 4/4 and every family >=3/4. It is a research decision rule, not a statistical significance claim.

## Operational interruption

The original seed-9102 test was interrupted by SSH monitoring failures before any test data were exported or scored. Development data and frozen classifiers were retained. A documented replacement used seed 9103 with identical settings and success criteria. The recovery runner preserved partial logs and tolerated monitoring outages without restarting the job. See [recovery amendment](LEDGER_PROBE_RECOVERY.md). This replacement and the missing original test observations are part of the experiment history.

## Cost and ordering

Fresh test incidents averaged 8.80s (maximum 11.63s), including both arms and warmups. The extra checkout pair averaged 1.31s and the ledger pair 0.99s, including telemetry reads. Counts match, but upstream work and wall time intentionally differ.

| First extra probe | Repeated checkout: pool/delay | Direct ledger: pool/delay |
|---|---:|---:|
| checkout | 13/16 | 16/16 |
| ledger | 15/16 | 16/16 |

## Interpretation and next step

The extra probe passed the screening rule and warrants collecting trajectories with explicit probe target and concurrency actions. Train and evaluate a new Dreamer action contract on those trajectories, with an equally informed telemetry control and a separate frozen test. The existing workload-only model cannot predict this untrained intervention. This result does not yet establish that Dreamer adds value.

## Limits and verification

The test uses the same eight synthetic fault families and 3/9 concurrency levels as development, with independently sampled severities. All requests succeeded; this is performance-fault triage, not outages or unseen-fault generalization. Both arms share an incident and execute sequentially; balanced order reduces but cannot eliminate state carryover. Declared client concurrency is not a guarantee of simultaneous arrival at the service. The control uses the same ledger-latency feature shape; additional upstream metrics from the repeated probe were not added, so this is a matched-feature acquisition comparison, not the strongest possible telemetry classifier.

Verification checked source hashes, balanced incidents, successful requests and per-service counts, fault-duration limits, equal request budgets, label isolation, training-only selection, frozen parameters before test collection, predictions and metrics. Three focused tests cover counter exclusion, measurement failure handling, and monitor-timeout recovery.

All work remained in systems-world-model-lab. The collector mounted no model or results volume and held no Kubernetes API credential. Inference was temporarily paused for quota, then restored to its original template and replica count. Temporary jobs and the collector ConfigMap were removed; final namespace health was checked.
