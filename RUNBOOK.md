# Reproduce the bounded POC

This runbook affects only `systems-world-model-lab`. It is not an AutoSRE runbook.

## Source and dependencies

```sh
git clone https://github.com/danijar/dreamerv3.git upstream
git -C upstream checkout e01491fad6434b2245a3b8ca201dd7faedcc458c
python3 scripts/bundle.py upstream /tmp/swm-source.tar.gz
```

The source bundle contains the unmodified upstream packages and license, plus
our environment and runner. `requirements-cpu.txt` pins direct dependencies;
`reports/evidence/pilot/requirements-resolved.txt` captures the actual full run.
Use that resolved file for exact environment reproduction. No CUDA or AMD GPU
runtime is used. The deployment image digest is recorded with the evidence.

## Isolation-first deployment

Confirm the Kubernetes context points to `amd-halo`, free capacity is sufficient,
and the namespace is absent or owned by this project. Record unrelated pod health
before deployment. The manifest imposes CPU/memory caps, a 10 GiB PVC request,
non-root execution, no service-account token, and default-deny ingress/egress.
Local-path storage requests are not a hard per-directory disk quota; monitor disk
usage and retain the job deadline rather than claiming a hard 10 GiB filesystem cap.

Create the Namespace/quota/policies/PVC from `deploy/lab.yaml`, create the source
ConfigMap below, then apply its Job (the initial Job can wait for the ConfigMap):

```sh
kubectl apply -f deploy/lab.yaml
kubectl -n systems-world-model-lab create configmap experiment-source \
  --from-file=source.tar.gz=/tmp/swm-source.tar.gz
kubectl -n systems-world-model-lab logs -f job/dreamer-poc-v1
```

After installation logs show `DEPENDENCIES_AND_TESTS_READY`, remove temporary
package-download access. This does not affect localhost communication:

```sh
kubectl -n systems-world-model-lab delete networkpolicy dependency-downloads
```

Wait for the pilot to complete before applying `deploy/main-job.yaml`. Never run
the two training jobs concurrently. The main job reuses the dependency environment
on the PVC but starts new model weights and uses fresh evaluation seeds.

```sh
kubectl -n systems-world-model-lab wait --for=condition=complete job/dreamer-poc-v1 --timeout=30m
kubectl apply -f deploy/main-job.yaml
```

Review `reports/PROTOCOL.md` before changing budgets or interpreting results.
There is no LLM service, public endpoint, or live production fault injection.

## Evidence and local execution

The PVC stores `/results/results-*.json` for the pilot and
`/results/main/results-*.json` for the main run. Each contains every evaluation
episode. Logs, configurations, dependency versions, and checkpoints are retained.
A temporary read-only results pod is described in `deploy/results-reader.yaml`;
use it after training finishes, then delete only that pod.

Use a fresh output directory for every experiment; the runner refuses existing
per-seed outputs to prevent accidental checkpoint reuse. Existing Kubernetes Job
names also need a new experiment suffix for a new run.

To run without Kubernetes in a separate environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt
export DREAMER_SOURCE="$PWD/upstream"
export PYTHONPATH="$PWD:$DREAMER_SOURCE"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest tests -q
.venv/bin/python -m experiment.run --seed 0 --steps 50000 --out artifacts/run
```

For an actual learned diagnostic trace, run `python -m experiment.diagnose` with
`--config` pointing to the saved train configuration and `--checkpoint` pointing
to a saved checkpoint directory. The trace is synthetic, not a live repair.

Aggregate and render:

```sh
python3 scripts/summarize.py reports/evidence/main > reports/evidence/main-summary.json
python3 scripts/paper.py
```

Paper generation requires ReportLab. Render the PDF with `pdftoppm` and inspect
all pages. Do not substitute pilot data for missing main results.

## Stop and retain data

Active jobs have a two-hour deadline. If intervention is needed, delete only the
named experiment job in the confirmed lab namespace. Completed jobs use no CPU.
Keep the PVC for reproducibility; export evidence before any explicit deletion.
Do not change cluster settings or workloads in any other namespace.

## Frozen follow-up and confirmation

The selected development configuration uses 100,000 steps,
`--abstention-penalty 2.0 --inspection-coverage 0.5`, and the native
16-particle inference wrapper. Original evaluation rewards are unchanged.
See `reports/FOLLOWUP_PROTOCOL.md`; development success is not confirmation.

`deploy/confirmation-quota.yaml` retains the four-CPU / 6 GiB limits and permits
three pods. `deploy/confirmation.yaml` assigns each independent seed 1333m CPU
and 2 GiB RAM, separate source/replay/checkpoint directories, and fresh evaluation
seeds. It uses the frozen `experiment-source-v7` bundle and existing dependency
venv. Wait for prior jobs/collectors to finish before applying these manifests.
For a new reproduction, create that ConfigMap from the desired pinned bundle and
use fresh job names/output directories; never overwrite recorded experiments.

The confirmation evidence retains both sampled native-actor results and the
particle-vote controller. Validate the gate with `scripts/summarize_followup.py`;
only seeds 10, 11 and 12 together can pass the frozen confirmation gate.

For a human-readable demonstration from a saved checkpoint:

```sh
python -m experiment.diagnose --config /results/confirmation-v4/train-10/config.yaml \
  --checkpoint /results/confirmation-v4/final-10 --controller particles --all-faults
```

This is simulated diagnosis, not a production incident or remediation service.
The native model weights are unchanged by inference voting.

## Finished job cleanup

Every checked-in Job manifest and `scripts/micro_job.py` sets
`spec.ttlSecondsAfterFinished: 86400`. Kubernetes removes successful and failed
Jobs and their dependent pods approximately 24 hours after completion. Running
Jobs are unaffected. New Job templates must retain this field; this is a
repository default, not a cluster-wide admission policy.

Export pod logs and job status into the experiment evidence before that window
expires. Checkpoints and experiment artifacts on the independently managed
`results` PVC survive Job deletion; pod logs do not. Reader/staging Pods are
not Jobs and must still be removed explicitly by their orchestration scripts.

`scripts/cleanup_finished_jobs.py` archives terminal-job status and available
pod logs locally. With `--apply`, it rechecks job identity/terminal state and
deletes only those archived jobs in `systems-world-model-lab`. It verifies
that deployment specifications and PVC identities are unchanged.
