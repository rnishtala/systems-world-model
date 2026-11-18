# Trace-only temporal diagnosis: workload-transfer results

2026-11-18. [Frozen protocol](LAB_TEMPORAL_PROTOCOL.md).
[Evidence](evidence/lab-temporal-v1/evaluation/summary.json).

## Experiment

A new upstream DreamerV3 seed 61 was trained from scratch on 80 measured training incidents. Direct CPU, pool-wait and database-duration counters were excluded from all classifier inputs. Each incident exposed four independent latency/error/load/trace windows; faults persisted while concurrent load alternated between 4 and 6. After every readout was frozen, 80 fresh incidents used alternating load 7 and 9, with the same fault-severity ranges. Both phases were balanced within each of the eight known fault families. This tests workload transfer, not unseen causes.

Training completed 100,000 steps in 17.2 minutes.
All methods used the same four observations. Latest-snapshot and full-history telemetry classifiers received the same linear/RBF candidate grid as the Dreamer-state classifier. Selection used training-only folds; these are not independent validation of the world model.

## Fresh evaluation

| Method | Exact diagnoses | 95% Wilson interval | Reliability gate |
|---|---:|---:|---|
| latent | 54/80 | 56.6%–76.8% | Fail |
| history | 49/80 | 50.3%–71.2% | Fail |
| latest | 41/80 | 40.5%–61.9% | Fail |
| auxiliary | 53/80 | 55.4%–75.7% | Fail |
| shuffled_latent | 3/80 | 1.3%–10.5% | Fail |

Exploratory added-value gate: **failed**. It requires reliability and four more correct diagnoses than the stronger telemetry baseline.

Against history: Dreamer alone correct on 21, baseline alone correct on 16; two-sided exact paired p=0.5114.
Against latest: Dreamer alone correct on 20, baseline alone correct on 7; two-sided exact paired p=0.01916.

| Fault family | Dreamer state | History | Latest |
|---|---:|---:|---:|
| Healthy | 10/10 | 10/10 | 10/10 |
| CPU | 10/10 | 3/10 | 4/10 |
| Pool | 4/10 | 0/10 | 4/10 |
| Delay | 1/10 | 10/10 | 0/10 |
| CPU + delay | 6/10 | 7/10 | 0/10 |
| CPU + pool | 8/10 | 3/10 | 5/10 |
| Pool + delay | 10/10 | 10/10 | 10/10 |
| All three | 5/10 | 6/10 | 8/10 |

Checkout latency exceeded the fixed 600 ms normalization ceiling in 20/320 training windows and 108/320 test windows. Both methods received the same capped values. Ledger and downstream trace timing clipped in 1 and 3 test windows respectively. This is a measured information-loss limitation, not proof that it caused all diagnosis errors.

## Interpretation and next step

The new experiment did not meet the reliability gate. A small aggregate lead is not sufficient evidence of a useful world-model advantage; per-family errors and the paired comparison above are the relevant limits. This harder task changes the observation view and test workload relative to earlier experiments, so their accuracy numbers are not directly comparable.

Next, revise the latency scaling using development data, broaden training workload coverage, and select readouts using development splits separated by operating regime, then evaluate on newly collected held-out sequences. The current random incident folds cannot measure extrapolation to a wholly different load range. Preserve this test batch as evaluation-only. There is no evidence here that a larger or more complex model is the necessary next change.

## Limits and operational outcome

One model seed and 80 controlled test incidents cannot establish general superiority. No action-selection, counterfactual-remediation, or early-stopping benefit was tested. The raw history baseline controls for simply retaining past measurements. Fault labels were excluded from Dreamer encoder/decoder, with an actual upstream policy-invariance audit. Local verification reproduced frozen readouts, test predictions and collection boundaries; latent features were not independently regenerated locally.

Checkpoint and readouts remained unchanged during evaluation. The previous inference template and replica count were restored, and the temporary Job/stager were removed. No model was promoted. The Kubernetes job completed in 26m28s. Bulk export subsequently failed; a temporary read-only exporter recovered all evidence in 32 KiB chunks and checked file hashes. This did not rerun training or evaluation. Future runs use the bounded exporter by default.
