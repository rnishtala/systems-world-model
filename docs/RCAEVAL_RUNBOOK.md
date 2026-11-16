# Isolated RCAEval experiment

All Kubernetes commands must include `-n systems-world-model-lab` and use the
existing lab kubeconfig. Do not apply the original RCAEval application stacks.

## Data and checks

Use Python 3.12 in a separate local venv; Python 3.14 does not support the pinned
NumPy wheel. Install `benchmark/requirements-prepare.txt`, then run:

```sh
python -m benchmark.prepare
python -m unittest tests.test_rcaeval -v
```

Data lives in ignored `data/rcaeval-v1/`. Preparation uses a pinned public dataset
revision, records SHA256s, excludes invalid timestamp windows, and anonymizes
case paths. Training can receive only `blind/` and training-only service labels.
Do not upload `answers.json`, `cases.parquet` or original case paths to training.
The full answer file is for a separate scorer after predictions are frozen.

## Cluster execution

The stage pod (`deploy/rcaeval-stage-v1.json`) uses 100m CPU and at most 768Mi.
Upload a tar containing `benchmark/`, `experiment/`, `blind/` and
`train-answers.json` to `/results/rcaeval-v1`. It uses the existing PVC Python
environment and pinned Dreamer source, with no dependency installation.
The stage pod runs the baseline via `python -m benchmark.evaluate --data blind
--out baseline` from that directory. Export predictions before deleting it.

Record the current inference replica count. Pause only `swm-inference`, wait for
quota to be released, and create `deploy/rcaeval-train-v1.json`. It is bounded to
2 CPUs, 4Gi memory and one hour. Always delete the experiment job and restore
the original inference replica count, including on failure. Check rollout
readiness. Then use the stage pod to export model metadata and predictions.
Do not reuse an existing training output; retain failures and choose a new run
name/path for a corrected attempt.

Both manifests use the `rcaeval-offline` role, which matches no egress-allowing
policy; existing namespace default deny applies. No service account token,
privilege escalation, host mounts or host networking. Namespace quota is not
changed. The stage pod deadline is 30 minutes. Model files remain under the
separate lab PVC prefix; the existing serving package is never overwritten.

## Interpretation

Read the frozen protocol before scoring. `benchmark.evaluate.score` computes
service AC@k and Avg@5 using exact root-cause labels. The positive robust-shift
baseline is BARO-style, not an exact official baseline reproduction. Runtime
excludes the fixed five-minute telemetry collection window. A single seed and
25 test cases give an initial signal, not a production reliability claim.

Sources: https://arxiv.org/abs/2412.17015 ,
https://github.com/phamquiluan/RCAEval ,
https://huggingface.co/datasets/phamquiluan/RCAEval .
Dataset card declares MIT. Retain attribution and the original dataset revision.
