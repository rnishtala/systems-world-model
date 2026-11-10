# Auxiliary mechanism supervision — development experiment

Frozen before training: upstream DreamerV3 e01491f, original 140 calibration incidents only, seed 21, 100,000 environment steps, unchanged size1m configuration. Three independent sigmoid outputs share Dreamer’s posterior latent state. Binary cross-entropy weight 1.0 is added to the original objective, only after all three inspections and before terminal diagnosis. Labels are excluded from both encoder and decoder; inference supplies a constant placeholder, removed before policy computation. This is supervised world-model training, not an LLM fine-tune or unmodified Dreamer.

Fixed inference: inspect CPU, pool and trace; average probabilities over 16 posterior particles; threshold each at 0.5. Preserve the original reward-based diagnosis as a comparator. Fit the frozen ridge telemetry baseline (penalty 1) on the same 140 training incidents before testing. No tuning against prior held-out tests. Exploratory latent and shuffled-label ridge probes use the same fixed penalty and are fitted before fresh collection; their role is diagnostic, not candidate selection.

First verify actual upstream training, target exclusion, checkpoint round-trip and calibration diagnosis. Only then collect a fresh 80-incident batch (scenario seed 6500, ten per family, increased load, including the unseen triple). Require 10/10 healthy and at least 9/10 in every fault family. One development seed cannot establish reliability or causality; successful results require independent confirmation and an otherwise identical no-supervision ablation.

Keep the default deployment and earlier checkpoints unchanged. Training runs only in systems-world-model-lab within the existing 4 CPU / 6 Gi quota, with lab deployments temporarily stopped and always restored. Fresh test data must never enter training.

## Connectivity amendment before scoring

The seed-6500 collection was interrupted by monitor cleanup after an SSH subprocess timeout. It is an incomplete infrastructure run, retained without an accuracy score. No model or threshold was selected from it. Repeat the entire fresh batch at seed 6501 with the same frozen checkpoint, comparators, 80-incident design and acceptance gates. Catch subprocess timeouts explicitly and preserve running bounded jobs when monitoring connectivity is unknown.
