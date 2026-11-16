# Fresh-collection confirmation protocol

Frozen before downloading or scoring RE2-OB telemetry. Prior RE1-OB validation is development evidence only.

- Dataset: RCAEval RE2-OB, pinned revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`; metrics only, expected 90 incidents. Repetitions 1–2 train; repetition 3 test. Exclude incidents with fewer than ten valid timestamps on either side of the published injection time, before model predictions. Record every exclusion.
- Transfer the fixed training recipe, not the previous checkpoint. This is a fresh collection of the same application, not an unseen-system or zero-shot test. No validation split, hyperparameter search, or test-guided retries.
- DreamerV3: upstream pinned revision, size1m, 32,000 environment steps, seeds42/43, auxiliary root-service cross entropy weight1. Twenty 30-second telemetry frames; train-only metric vocabulary; existing per-incident pre-onset normalization.
- Primary readout: post-onset pooled full state, 16 posterior particles, ridge100. Raw sequence ridge1 is a secondary comparator. No readout menu on confirmation data.
- Supervised non-world-model baseline: flattened ordered sequence (including masks), one ReLU hidden layer of128 units, softmax service output, AdamW lr0.001/weight decay0.0001, 32,000 minibatch updates of16, seeds42/43. Same train incidents, labels, inputs and service vocabulary. Capacity and optimizer work are not matched; 32k Dreamer environment steps are not 32k neural optimizer updates. This comparison cannot isolate the dynamics objective causally.
- Test labels remain local, outside the uploaded trainer/inference bundle. Train labels must exactly match training IDs. Freeze all method predictions and hashes before opening test answers for scoring. Report service top1/top3, paired wins/losses, each seed separately, and per-fault results. Two seeds share the same test incidents.
- No test-driven tuning after results. Stronger evidence would require additional systems, more incidents, and a matched-architecture dynamics ablation.
- Namespace systems-world-model-lab only. Bounded offline job, existing quota and default-deny networking. Temporarily pause only the inference deployment to fit quota, then restore its unchanged template. Do not promote a new serving model.
