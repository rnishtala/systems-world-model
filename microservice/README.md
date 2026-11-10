# Standalone microservice experiment

All resources belong to `systems-world-model-lab`; all code and evidence belong
to this repository. Do not run AutoSRE/BearBank scenario or GitOps scripts.

The live call chain is checkout -> payment -> ledger -> PostgreSQL. The workload
uses real requests and real connection occupancy, with a synthetic database
query hold. It does not process real financial transactions.

`service.py` implements bounded fault leases and measured telemetry. Port 8080
serves requests/metrics; port 9090 is available only to the experiment runner.
`collect.py` records training trajectories. `environment.py` exposes a fast
empirical training environment. `train.py` reuses pinned upstream DreamerV3.
`agent.py` loads fixed checkpoints in a separate inference pod, with neither
control access nor the calibration dataset mounted. `evaluate.py` tests new live
incidents and records sensor forecasts before inspections. `triage.py` diagnoses
the current estate without selecting faults or reading truth labels.

Manifest generators are under `scripts/micro_*.py`. Their namespace is fixed;
CPU/memory caps remain four CPUs / 6 GiB. Stop these four service deployments and swm-inference
before allocating the full quota to training; resume them before live evaluation.
The PostgreSQL lab storage is ephemeral; checkpoints/data stay on the existing
lab PVC. No other namespace, repository, database, or credentials are used.

Dependency delivery is offline: install `microservice/requirements.txt` into a
scratch directory, then package it with `scripts/micro_bundle.py`. Create the
binary ConfigMap directly (`kubectl create configmap ... --from-file=...`), not
client-side apply, which duplicates a large payload in an annotation. Source
bundles are versioned; never replace evidence or checkpoints in place.

Acceptance criteria and dataset separation are in
[the protocol](../reports/MICROSERVICE_PROTOCOL.md). Deployment or replay accuracy
alone is not live success. Do not report a confirmation gate before all reserved
incident evaluations are complete. Forecast accuracy and triage accuracy are
separate results.

After a model is deployed, the read-only investigation command is:

```sh
kubectl --kubeconfig="$HOME/.kube/graphrobin.yaml" -n systems-world-model-lab \
  exec deployment/swm-inference -- \
  /results/venv/bin/python -m microservice.triage --seed 10 --concurrency 6
```

Use seed 0 for the development model; seeds 10/11/12 only after confirmation
models have been trained and deployed. The command issues bounded synthetic
request traffic to the lab, reads measured telemetry and uses the deployed controller
for its next check. It does not inject faults, change configuration or repair the
system. An idle or unavailable service must not silently become a successful
healthy diagnosis.

Controllers: `native` uses the learned actor with 16-particle voting.
`inspect-first` fixes three inspections (CPU, pool, trace), then ranks diagnoses
with imagined rewards from the trained world model. This mode is a harness
change, not retraining or learned inspection selection. Responses identify the
controller, and live traces retain it. Neither mode has established reliable
unknown-fault detection. `evaluate --unknown` separately tests combined CPU and
ledger delay, never included in calibration or known-class acceptance.
