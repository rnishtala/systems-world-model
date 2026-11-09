# Live microservice triage protocol

This experiment is limited to this repository and systems-world-model-lab.
The earlier queue-simulator result is historical; it is not evidence for this lab.

## Environment and boundaries

Checkout -> payment -> ledger -> PostgreSQL performs real HTTP calls and database
queries. A bounded query hold creates genuine connection-pool occupancy; this is
a synthetic banking-style workload, not an implementation of banking business
transactions. Independent control endpoints inject leased CPU work, pool limits
or ledger delay. No control parameters, scenario IDs or truth labels enter the
model's observation contract. Agent network policy blocks control port 9090.

All application flows remain inside the namespace. The only cross-namespace
network exception is DNS. The agent mounts only checkpoints, the pinned upstream
source and the retained Python runtime, read-only; it does not mount calibration
data. No pod receives Kubernetes API credentials or host mounts. CPU/memory caps
remain four CPUs / 6 GiB in aggregate. Pod quota increases to eight for the lab.
Services pause during training and resume for live evaluation. Faults expire
within 90 seconds and the experiment runner resets them on completion/failure.

## Data and learning

Calibration v1a: 80 real incidents (20 per class), shuffled with seed 1200,
concurrency 4–6, four measurement frames per incident. Classes: healthy,
payment CPU contention, ledger connection-pool contention, ledger delay.
The first collection attempt failed at startup and is retained separately.
Bounded readiness retries preceded the successful collection.

Training reuses the pinned DreamerV3 implementation, not the old queue checkpoint.
The new contract has 14 observations and nine actions. A fast empirical training
environment resamples complete measured trajectories, advances their frames,
and reveals selected sensors. It is a training surrogate, not a live service
and not an independent physics simulator. 50% of training episodes get three
randomly ordered checks followed by a random diagnosis; no expert answers are
provided. Train 100,000 steps per development seed, using the existing small CPU
configuration. Reward: -0.05/action, -0.15 extra per inspection, +3/-3 correct/wrong
diagnosis, -2 unresolved. Preserve unsuccessful trials and source hashes.

## What counts as learned system behavior

The native Dreamer actor, with 16-particle voting, must choose inspections and
identify the faulty service plus mechanism in new live incidents. For each
selected inspection, also record the world model's predicted next sensor values
BEFORE making that inspection. Compare these predictions with actual measurements
and with unconditional per-sensor means computed from calibration data only.
Record imagined diagnostic rewards separately. Accurate classification alone is
not a claim that the model understands arbitrary causal interventions or repairs.

## Development and frozen confirmation

Use a small development live batch (six incidents/class, scenario seed 2200)
and an increased-load/severity development batch (seed 2300) to find integration
and transfer problems. These are not confirmation results. Calibration and all
controller changes must finish before freezing a new confirmation configuration.

Then train fresh seeds 10, 11 and 12 without tuning against confirmation results.
Use 30 incidents/class at held-out loads and severities, scenario seed 4200.
All models may inspect the same physical incident, in shuffled order: report
120 physical incidents and 360 model-incident evaluations, not 360 independent
physical incidents. A topology-aware checklist gets the same inspection access.
Its initial thresholds are fixed before development evaluation; any tuning is
recorded and frozen before confirmation. Inference compilation happens before
injection and is excluded from diagnosis latency.

Per-seed triage gate:
- At least 90% exact diagnosis in each known fault class, including the cascade.
- At most 5% false fault diagnoses for healthy/increased-traffic incidents.
- At most three inspection actions on average; median diagnosis below 60 seconds.
- Abstentions count as unsuccessful known-class diagnoses.

Separate predictive-evidence gate: at least 30 selected-inspection forecasts per
seed, with pooled sensor mean-square error below the calibration-mean baseline.
Report per-sensor performance as well; this weak baseline is a first check of
learned predictive structure, not proof of a general systems dynamics model.
Do not declare this gate passed solely because diagnosis passed.

Include a separate combined/unfamiliar-fault test and report escalation and
confident misclassification. Unknown-fault handling is not established by the
known-class gate. No LLM, automatic remediation, Slack or GitHub incident workflow
is part of this stage.
