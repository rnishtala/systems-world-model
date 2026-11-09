# Systems World Model

[One-page research proposal](output/pdf/systems-world-model-proposal.pdf)

A standalone, executable DreamerV3 experiment for active systems diagnosis.
The first POC trains the project-linked DreamerV3 implementation from scratch
on a partially observed queue simulator. It runs on **AMD Halo, CPU only**, in
its own Kubernetes namespace: `systems-world-model-lab`.

## Confirmed synthetic POC result

**Passed the frozen confirmation gate on 9 November 2026.** Three freshly trained
models (seeds 10, 11, 12) each achieved **100% accuracy at both familiar and shifted
load**, with **one inspection per incident and zero unresolved cases**. That is
900 held-out incidents in total. All three saved-model demonstration cases also
passed. The three training/evaluation jobs completed in 26.7 minutes elapsed.

| Confirmation method | Familiar accuracy | Shifted accuracy | Inspections (familiar) |
|---|---:|---:|---:|
| Native sampled Dreamer actor | 90.0% | 93.8% | 0.99 |
| Native actor with 16-particle voting | 100.0% | 100.0% | 1.00 |
| Adaptive checklist | 100.0% | 100.0% | 1.67 |
| Fixed checklist | 100.0% | 100.0% | 2.00 |

The confirmed configuration trains from scratch for 100,000 steps, uses a higher
training abstention cost and training-only inspection coverage, and votes across
16 native latent-state/actor samples during inference. No expert diagnosis rules
or hidden fault labels are supplied at inference. Voting changes inference, not
model weights. This small synthetic result does **not** establish real-system
transfer, superiority to optimal expert rules, or LLM-assisted repair.

[Read the five-page follow-up research report](output/pdf/dreamerv3-triage-followup.pdf).
The [confirmation gate](reports/evidence/followup/confirmation-particle-gate.json),
[three-incident trace](reports/evidence/followup/confirmation/demo.json), failed
trials, and every seed's episode-level evidence are preserved. Runtime tests:
**10 passed**. Experiment processes and the temporary collector have stopped;
checkpoints remain in the isolated lab PVC.

## What is implemented

```mermaid
flowchart LR
    S[Queue simulator: hidden fault and load] -->|numeric observations + rewards| W[DreamerV3 world model]
    W -->|imagined trajectories| A[Actor and critic training]
    A --> P[Learned inspection / diagnosis policy]
    P -->|inspect, diagnose, or escalate| S
    P --> E[Held-out evaluation against checklists]
```

The encoder, RSSM, prediction heads, imagination training, actor and critic are
upstream code. The environment adapter and evaluation harness are ours. This
is model training for a new environment, not LLM fine-tuning. No BayesNet is used.

**Scope:** synthetic diagnosis only. No LLM calls, production telemetry, real
fault injection, code repairs, or real-service recovery are claimed. The later
LLM resolution architecture remains in [MVP_PLAN.md](MVP_PLAN.md).

## Original baseline experiment

Three 4,000-step pilot runs and three fresh 50,000-step main runs, with three
fault classes and seven actions. Each seed is evaluated on 150 familiar-load
and 150 shifted-load episodes per method. Fixed and adaptive checklists and a
uniform random policy use the same observation/action interface.

**Original baseline (9 November 2026):** the integration ran, but this initial learned
policy did not perform useful diagnosis. Across 450 episodes per load regime:

| Method | Familiar-load accuracy | Shifted-load accuracy | Familiar-load inspections |
|---|---:|---:|---:|
| DreamerV3 | 0.2% | 0.2% | 0.00 |
| Adaptive checklist | 100.0% | 100.0% | 1.67 |
| Random | 26.9% | 24.0% | 0.48 |

Dreamer escalated 99.3% / 99.6% of familiar / shifted episodes. Training took
18.54 minutes total for the three main seeds. The 4,000-step pilot had 25.6%
familiar-load accuracy; more training converged toward abstention. This is a
negative learning result, not evidence that the method is effective in real triage.
The follow-up below tests abstention reward and inspection exploration, retaining
this baseline as historical evidence.

[Read the four-page research report](output/pdf/dreamerv3-system-triage-poc.pdf).
Episode-level results, configurations and logs are under `reports/evidence/`. See [the protocol](reports/PROTOCOL.md) for budgets and limits;
all seeds, including unsuccessful policies, are retained.

## Follow-up experiments

These are development results (one seed, 150 episodes per load regime), not the
three-seed confirmation required by the [follow-up protocol](reports/FOLLOWUP_PROTOCOL.md).

| Training change | Familiar accuracy | Shifted accuracy | Observed behavior |
|---|---:|---:|---|
| Escalation penalty only | 32.0% | 34.0% | Guessing without inspections |
| Inspection coverage only | 0.0% | 0.0% | Escalates despite diagnostic experience |
| Combined correction, native actor | 93.3% | 94.7% | About one inspection |
| Combined correction, 16-particle native actor vote | 100.0% | 100.0% | One inspection |

A separate 90-episode probe **forced both inspections**. The learned world model
ranked the correct diagnosis first in 76.7% of cases, but the actor made no correct
diagnoses. Predicted rewards were still too pessimistic: this probe is not evidence
of autonomous triage. After the combined correction and 100,000 training steps,
the same probe reached 100% world-model ranking accuracy (reward RMSE 0.103).

The particle controller keeps 16 latent-state/actor samples of **one model**,
executes their majority action once, and feeds that action back to every sample.
It adds no diagnosis rules or extra sensor observations. The subsequent frozen
confirmation used seeds 10, 11, 12 and untouched evaluation streams; all three
passed, as reported above. No tuning was performed against confirmation results.

## Run and inspect

- [RUNBOOK.md](RUNBOOK.md): exact upstream pin, dependencies, deployment and reproduction.
- [UPSTREAM.md](UPSTREAM.md): source reuse and licensing.
- `experiment/environment.py`: bounded partially observed simulator.
- `experiment/run.py`: real upstream training and held-out evaluation.
- `experiment/diagnose.py`: load a saved checkpoint and print a simulated incident trace.
- `scripts/followup_paper.py`: regenerate the current research PDF from measured evidence.
- `scripts/proposal.py`: regenerate the one-page proposal.
- `scripts/paper.py`: regenerate the historical baseline report.

The lab has a four-core / 6 GiB memory limit, default-deny networking after
installation, no Kubernetes API credentials, and no host mounts. Training jobs
finish automatically; checkpoints stay in the lab PVC. Namespace boundaries do
not eliminate shared-host contention. Readiness/restart evidence and existing
Argo CD instability are recorded explicitly.

This is independent of AutoSRE and is not a production remediation agent.
