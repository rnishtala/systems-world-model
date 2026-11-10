# Compositional diagnosis: development result

**The new Dreamer candidate failed: 49/80 exact diagnoses (61.25%).** A frozen,
lightweight learned comparator scored 76/80 (95%) on the identical observations.
Neither passed all prespecified family gates. The previous inspect-first
controller was restored; this result is not promoted as reliable triage.

| Incident family | Dreamer seed 20 | Independent binary trees |
|---|---:|---:|
| Healthy | 10/10 | 10/10 |
| CPU | 10/10 | 10/10 |
| Pool | 9/10 | 9/10 |
| Delay | 10/10 | 10/10 |
| CPU + delay | 0/10 | 10/10 |
| CPU + pool | 0/10 | 9/10 |
| Pool + delay | 10/10 | 10/10 |
| CPU + pool + delay, unseen in training | 0/10 | 8/10 |

Dreamer detected CPU in all 40 CPU-bearing incidents but recalled only 50% of
pool and delay mechanisms. It made no healthy false alarms and no escalations.
Median end-to-end diagnosis was 3.78 seconds, with three inspections. The binary
trees were evaluated offline, so their classification time is **not** a comparable
end-to-end latency measurement.

## What changed

Actual upstream DreamerV3 was retrained from scratch with eight fault-set answers
and additive rewards for the three independent mechanism decisions. This changes
the task adapter and weights, not an LLM or a Bayesian model. The model still
uses one learned reward function to rank eight imagined terminal actions; it
has no separate supervised mechanism heads.

Calibration contained 140 real HTTP/PostgreSQL incidents, 20 per healthy/single/
pair family, alternating ordinary and increased load/severity. The live test
contained 80 fresh incidents at increased load, including ten triples absent
from calibration. Increased load and pairs were represented in training, so this
is not a matched comparison with the earlier single-fault studies.

Training took 737 seconds (12m17s). The evaluation workflow stage lasted about
7m33s including job startup and tests. All 30 runtime regression tests passed.
The comparator used three depth-three binary trees, minimum leaf five, fitted on
calibration data before live evaluation. Training and comparator data hashes
match. No test outcomes selected the comparator settings.

## What the result establishes

- Pool-versus-delay separation worked in this small batch, including all ten
  pool-plus-delay incidents, but that is not fresh multi-seed confirmation.
- A broader answer vocabulary and combined-fault training did not make Dreamer
  compose diagnoses when CPU contention was present.
- The comparator recovered most of these combinations from the same telemetry.
  Missing signals alone therefore cannot explain Dreamer's systematic failure.
- This experiment cannot yet distinguish information lost in the recurrent
  latent state from a weak learned reward readout. A **post-hoc**, unfitted
  additive projection of predicted rewards also scored 49/80; final argmax
  interactions alone do not explain the failure.
- Calibration latency clipped in 9/20 pool-plus-delay incidents. This limits
  observability but does not explain the main CPU-combination failure pattern.

## Next bounded experiment

Probe the frozen Dreamer latent state with three independent mechanism classifiers,
fitted only on training trajectories. Compare with raw-telemetry classifiers on
fresh held-out incidents. This locates whether the recurrent representation or
the diagnostic readout needs improvement before spending another training cycle.
Do not present the 95% baseline as a reliable replacement: it failed the triple
family gate, and this is one development batch with only ten cases per family.

## Evidence and isolation

Protocol: [MECHANISMS_PROTOCOL.md](MECHANISMS_PROTOCOL.md). Raw calibration and
live traces, fitted baseline, source hashes, checkpoint metadata, logs, per-family
scores and restoration record: [evidence/mechanisms](evidence/mechanisms).
The PDF is [the compositional diagnosis research note](../output/pdf/dreamerv3-compositional-diagnosis.pdf).

All writes and artifacts are in this standalone repository. All cluster changes
remained within `systems-world-model-lab` under the existing 4 CPU/6 GiB cap.
The five lab deployments are restored; the experiment jobs were removed after
exporting their evidence. Previous checkpoints and historical results remain.
