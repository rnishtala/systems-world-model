# Partial-telemetry matched-objective experiment

Frozen before RE2-SS telemetry is downloaded or test predictions are scored.

## Question and design

Does adding DreamerV3 world-model training improve service localization when observations are intermittently missing, compared with the same recurrent architecture trained only on diagnosis labels?

Use previously unused RCAEval RE2-SS (Sock Shop), pinned dataset revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`. Metrics only; expected 90 incidents. Repetitions 1–2 train, 3 held-out test (60/30 expected). Same published-onset window and predeclared timestamp exclusions as prior confirmation. No test-guided tuning or retries. The application is new to this experiment, but each candidate trains on Sock Shop: this is not zero-shot cross-system transfer.

Normalize each metric using only its own pre-onset data. Keep all ten pre-onset frames. For each of ten post-onset frames, deterministically hide ceil(half) of the service groups in the train-derived vocabulary, setting both their values and observation masks to zero. Use SHA256 of fixed mask seed 20261117, anonymized incident ID, frame number, and service name to order hidden groups. The pattern does not depend on labels or telemetry values. Both models receive identical masked arrays. No full post-onset observations are training reconstruction targets. This tests robustness to intermittent telemetry loss, not denoising supervision, learned inspection selection or counterfactual intervention.

## Frozen models and inference

- Seeds 42/43, 32,000 environment steps each, original size1m recurrent stochastic state architecture, same replay stream configuration, optimizer and diagnosis head.
- World-model candidate: existing upstream Dreamer loss plus weight 1 post-onset service cross entropy.
- Matched supervised recurrent control: exactly the same forward recurrent architecture and initialization, but only service cross entropy contributes to optimization. Original reconstruction, dynamics/KL, reward, continuation, actor and critic objectives are disabled together. They may still be computed for metrics. This tests the combined Dreamer objective, not dynamics loss alone. Unused heads remain allocated and are not a capacity advantage for diagnosis.
- Audit the actual model with targets excluded from encoder/decoder/policy; check that opposite supplied labels cannot alter policy inference. For the control, verify numerically on a training batch that optimized loss equals diagnosis loss, and that removing the original loss removes its gradients from the objective.
- Primary: half-service masked telemetry; post-onset pooled full latent state, 16 posterior particles, ridge100 fitted only on training labels. Raw flattened masked telemetry ridge1 is another comparator.
- Secondary: complete telemetry with the same frozen models, fitting the same fixed readout on complete training telemetry. This measures input sensitivity, not a separately trained full-telemetry model. No selection between conditions.
- Test labels stay local; upload only train labels. Freeze predictions for every method, seed and condition before a separate local scorer reads test answers. Score service top1/top3 and paired errors by fault.

## Decision and limits

An encouraging primary result requires world-model top1 to exceed both the matched control and raw ridge by at least 3 of 30 incidents in each seed (10 percentage points if 30 cases remain). Otherwise do not claim demonstrated world-model value. Report exact paired counts and both seeds, not an inflated 60-case sample. This small benchmark gate is exploratory, not statistical proof or a deployment acceptance gate. Do not promote a serving model.

Use only systems-world-model-lab, existing quota and no-egress experiment pods. Four training runs are sequential. Temporarily pause its inference deployment to fit quota, restore the original replica count/template in a finally block, archive outputs, and delete the experiment job. Job TTL remains 24 hours as a fallback. Preserve all checkpoints and failure evidence. No work in other namespaces.
