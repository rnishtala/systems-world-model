# Compositional diagnosis development protocol

Prospective experiment after predictive-guard-v1 failed. No reliability claim yet.

## Hypothesis and change

The former four-way answer cannot describe simultaneous mechanisms. Keep the
same upstream DreamerV3 architecture, 14 telemetry features, three inspections,
and physical-v2 normalization. Retrain from scratch with eight diagnosis actions
representing independent CPU/pool/delay bits. Terminal reward is +1 per correct
presence/absence decision and -1 per incorrect decision, minus the action cost.
The learned reward model ranks the eight complete answers after three checks.
This is a new task adapter and checkpoint, not LLM fine-tuning or a Bayesian layer.
It is a discrete compositional reward objective, not separate Bayesian heads.

## Data and reserved evaluation

- Calibration: seed 6200, 20 incidents per family: healthy, three singles and
  three pairs (140 total). Alternate ordinary and shifted load/severity;
  four measured frames each. No triple-fault calibration incidents.
- Development model: seed 20, 100,000 steps, 50% random inspection/answer coverage.
- Development live test: seed 6300, 10 incidents per each of eight families,
  shifted load, including the previously unseen triple combination.
- Gate to justify fresh confirmation: at least 9/10 exact diagnosis sets per
  family; healthy 10/10; zero failed requests/expired fault leases. Report each
  mechanism's precision and recall, exact-set accuracy, and latency separately.
  With only ten cases per family, passing is exploratory, not reliability proof.
- Reserve scenario seeds 6400 and 6500 and model seeds 21/22 for a later,
  prospectively specified confirmation if development passes. No tuning on them.
- Do not reuse the old guard thresholds or score pair incidents as unknown here:
  pairs are explicitly in training. Only the triple composition is held out.

## Isolation and rollback

All workload changes stay in systems-world-model-lab. Keep the 4 CPU/6 GiB quota,
45-second injection leases, runner-only fault controls, and no service account
credentials. Stop inference during collection if capacity requires; stop the
five lab deployments only during training. Use versioned source/checkpoints.
Restore deploy/microservice-agent-confirmation.json after development evaluation
regardless of outcome. Preserve earlier evidence and report any regressions.

## Comparator addendum (frozen during training, before live evaluation)

Fit three independent binary classification trees to the same 140 calibration
incidents, maximum depth 3 and minimum leaf 5. Select splits by weighted binary
Gini impurity; ties predict absence. Use features 2..9 from the same fixed
CPU/pool/trace inspection sequence. No test outcomes select features or settings.
Score the fitted trees offline on the Dreamer live observation traces; no extra
injections are needed. Report exact sets and mechanism precision/recall. Their
classification-only time must not be compared with Dreamer's end-to-end latency.
This comparator tests whether a simpler model suffices for the injected faults.
