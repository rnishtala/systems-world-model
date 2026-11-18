# Fresh confirmation: development-selected check orders

**Prespecified confirmation gate: failed.**

Policies were fitted and frozen using scenario 8201 development data, then tested on 80 fresh scenario 8401 incidents (ten per known fault family). Neither policy changed during evaluation. This is a fresh confirmation of the fitted policies, separate from the preceding nested development result.

| Method | Frozen check order | Correct / 80 | Mean checks / 3 | Reliability gate |
|---|---|---:|---:|---|
| Dreamer state, stopping | pool → CPU → trace | 80 | 2.962 | Pass |
| Dreamer state, all checks | pool → CPU → trace | 80 | 3.000 | Pass |
| Telemetry history, stopping | trace → pool → CPU | 79 | 1.538 | Pass |
| Telemetry history, all checks | trace → pool → CPU | 79 | 3.000 | Pass |

## Prespecified criteria

- Pass: Reliability: healthy 10/10, other families ≥9/10.
- Pass: At most one fewer correct diagnosis than chosen-order full checking.
- Fail: Save ≥0.5 mean checks versus three checks.
- Fail: Save ≥0.5 mean checks versus telemetry stopping.

Paired correctness: Dreamer alone correct 1; telemetry alone correct 0.

## Per-family stopping accuracy

| Family | Dreamer / 10 | Telemetry / 10 |
|---|---:|---:|
| Healthy | 10 | 10 |
| CPU | 10 | 10 |
| Pool | 10 | 10 |
| Delay | 10 | 10 |
| CPU + delay | 10 | 10 |
| CPU + pool | 10 | 10 |
| Pool + delay | 10 | 10 |
| Triple | 10 | 9 |

## Interpretation and limits

The earlier development selection scored Dreamer 80/80 at 2.425 checks versus telemetry 78/80 at 2.975. Those numbers are not confirmation results. The final fit selected one order per method using all development cases; it did not choose the best order from this fresh batch. Readouts are supervised classifiers over a frozen Dreamer state or matched raw history. This does not demonstrate native Dreamer planning or isolate the causal benefit of world-model objectives.

Checks have unit cost. The experiment records complete trajectories and replays observation acquisition, so check savings are simulated diagnostic acquisitions, not measured wall-time savings. Reordering assumes passive inspections do not change system dynamics. This is one model and a small known-regime batch, not unseen-system or production reliability. No model was promoted.

## Verification and operations

Job duration: 7m38s; staging and restoration add wall time. An initial SSH transfer timeout occurred before job creation or any collection; its log is preserved. No collected incident was rerun or excluded. Seventeen targeted tests passed.

Verification reproduced final policy selection and every fitted coefficient from development data, then every fresh evaluation score and stopping decision. It checked source/checkpoint hashes, freeze-before-collection timestamps, clean measurement windows, distinct development/evaluation trajectories, observed-prefix boundaries and unchanged model/parameters. Inference was restored with its original template and replica count; temporary resources were removed. All operations stayed in systems-world-model-lab.

[Development results](LAB_CHECK_ORDER_RESULTS.md) · [Frozen confirmation protocol](LAB_ORDER_CONFIRM_PROTOCOL.md) · [Verification](evidence/lab-order-confirm-v1/verification.json) · [Detailed results](evidence/lab-order-confirm-v1/evaluation/summary.json)

## Decision and next step

Do not promote this candidate or claim a world-model efficiency advantage. Telemetry's frozen trace-first policy used 1.5375 checks versus 2.9625 for Dreamer's pool-first policy: about 48% fewer checks, with one fewer correct diagnosis and both passing the preset reliability gate. These savings count diagnostic observations, not measured execution time.

The promising development advantage did not survive the final fit and fresh confirmation. Development selected different orders across folds, and the full-data fit selected yet another policy configuration. That motivates a development-only audit of order-selection and confidence stability before further live runs. It does not establish the cause of the reversal, and this confirmation batch must remain untouched for fitting. Retain the strong telemetry baseline; a world-model contribution must demonstrate value beyond it.
