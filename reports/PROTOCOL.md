# Experiment protocol

The experiment is a deliberately small, synthetic partial-observation diagnostic
task, not a deployment of real faults into production services.

- Upstream: danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c.
- Reuse: encoder, RSSM, prediction heads, imagined trajectories, actor and critic;
  upstream files are unmodified. The adapter supplies a new environment factory.
- Model: upstream size1m configuration, 638,671 actual trainable parameters;
  numeric observations only; CPU, float32, batch 8 x 16, train ratio 16.
- Training faults: balanced in expectation between healthy, CPU bottleneck,
  and downstream slowdown. Passive observations for the two faults have identical
  distributions. Inspection results are noisy but strongly separable.
- Actions: wait, inspect CPU, inspect dependency, three diagnoses, escalate.
- Reward: -0.05 per action; additional -0.15 per inspection; +2 correct or -2
  wrong diagnosis; -0.5 escalation or timeout. Episodes end within 6 actions.
- Pilot: 4,000 environment steps for seeds 0, 1, 2. First pilot accuracy showed
  undertraining; this informed one larger compute-budget choice.
- Main experiment: 50,000 environment steps per seed from scratch, same seeds.
  No architectural or reward tuning based on final evaluation.
- Evaluation: 150 balanced episodes per seed at training-range load and 150 at
  higher load; fresh RNG seeds, 1000000+seed and 1100000+seed for the main run.
  Fixed checklist, adaptive checklist, and uniform random baselines. Baselines
  receive the same sensors and actions. Their simulator RNG consumption differs,
  so these are distribution-matched rather than exact paired trajectories.
- No LLM participates. No production telemetry is used. No code remediation or
  real-service transfer is evaluated. An LLM resolution stage remains future work.
- Action probabilities are not interpreted as calibrated root-cause probabilities.
- Preserve all seed results, including failures. Do not infer broad system-triage
  competence, causal repair prediction, or a production advantage from this task.
