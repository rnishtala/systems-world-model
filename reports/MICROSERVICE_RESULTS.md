# Live microservice experiment results

Later evidence: [the 10 November follow-up](UNKNOWN_FAULT_RESULTS.md) exposed
known-delay accuracy below target on a fresh stream. The results below remain
the historical frozen confirmation, not a universal reliability claim.

Known-case confirmation passed on 9 November 2026. This is a bounded experiment,
not a claim of general system understanding or production readiness.

## Result

Three fresh DreamerV3 models trained from real calibration trajectories diagnose
new live incidents through the inspect-first harness: three fixed inspections
(CPU, pool, trace), followed by learned imagined-reward ranking. There is no
expert diagnosis threshold in this controller. Native learned inspection choice
was unsuccessful and remains documented separately.

| Seed | Correct / physical incidents | Healthy false alarms | Median diagnosis | Forecast MSE / mean baseline |
|---|---:|---:|---:|---:|
| 10 | 119 / 120 | 0 / 30 | 1.64 s | 0.1981 / 0.2846 |
| 11 | 120 / 120 | 0 / 30 | 2.39 s | 0.2123 / 0.2853 |
| 12 | 119 / 120 | 0 / 30 | 2.44 s | 0.2015 / 0.2874 |

All three satisfy the frozen diagnosis and forecast gates. Thirty incidents per
class were shared across methods in shuffled order: 120 physical incidents,
360 model evaluations, 120 checklist evaluations. Each model made 360 forecasts.
These observations are paired, not 360 independent incidents. A perfect 30/30
class sample still has a 95% Wilson lower accuracy bound near 88.6%.

The checklist scored 119/120, using 2.25 checks and a 1.60-second median. It
produced one false alarm on healthy incident 116. The model comparison does not
establish statistical superiority or lower inspection cost.

## Failures and boundary

Original-feature native actor: 25% live development accuracy, no inspections.
Physical scaling with unchanged training budget/data: 75%, all six delay cases
missed. With all inspections forced on calibration data, the scaled world model
ranked 80/80 correctly; this was a diagnostic probe, not held-out evidence.
The same model with the inspect-first harness then achieved 48/48 on two fresh
live development batches, and forecasts beat the mean baseline in both.

In confirmation incident 53, seeds 10 and 12 classified ledger delay as pool
contention; seed 11 was correct. Their observed normalized pool waits were
approximately 0.157-0.159 versus 0.00013 for seed 11, as methods measure successive
batches. This is an observed difference, not proof of the errors' causal origin.

In eight separate combined CPU-plus-delay cases, both the development model and
checklist reported CPU and missed delay. No escalation occurred. The current
harness always ranks one of four known diagnoses; scores are not calibrated
confidence. Unknown faults, unseen topology, causal repair predictions, LLM
integration and automated recovery remain outside the demonstrated capability.

## Execution and verification

Fresh model training took 25m49s elapsed for all three concurrent jobs under the
unchanged four-CPU/six-GiB namespace cap. Live confirmation took 20m23s. Models
were compiled before fault injection. An orchestration parser initially expected
a List for one Kubernetes Job; fixing the monitor did not restart or alter the
ongoing live evaluation. Earlier startup and serialization failures are retained.

Runtime suite: 13 passed; two additional regression checks passed for live versus
training observation parity and rejection of failed request batches. Final source
syntax and diff checks passed. All 480 confirmation records have the reserved
scenario seed, shifted load, correct episode counts and bounded inspections.

The final inference pod can reach telemetry, but TCP probes confirm fault-control
and direct PostgreSQL ports are blocked. Kubernetes tokens and calibration data
are absent. A separate read-only healthy investigation passed with three checks
in 0.30 seconds. Five lab deployments remain ready; experimental jobs and the
temporary reader have finished. No other repository or namespace was changed.

## Reproduce / inspect

- `MICROSERVICE_PROTOCOL.md`: prospective criteria and explicit controller freeze.
- `evidence/microservice/confirmation-summary.json`: exact per-seed scores and intervals.
- `evidence/microservice/confirmation-live.jsonl`: observations, predictions, traces, truth.
- `evidence/microservice/confirmation-job-timings.json`: Kubernetes execution timestamps.
- `evidence/microservice/provenance.json`: source and immutable calibration hashes.
- `../microservice/README.md`: operating instructions and inference command.
- `../output/pdf/dreamerv3-live-microservice-study.pdf`: research report.

The end-to-end test operates real HTTP services and PostgreSQL, using bounded
synthetic requests and query holds. It is not the historical queue simulator,
and it is not an implementation of banking transaction business logic.
