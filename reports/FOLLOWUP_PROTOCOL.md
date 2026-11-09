# Follow-up protocol: from runnable integration to useful diagnosis

The first three main runs remain immutable negative evidence. This protocol is
prospective for follow-up evaluation, not a reinterpretation of their outcome.

## Success gate

Each of three fresh confirmation training seeds must achieve at least 95%
familiar-load accuracy, 90% shifted-load accuracy, at most two inspections per
episode on average in either regime, and at most 5% unresolved familiar-load
incidents. Use 150 balanced episodes per seed per regime. This is an empirical
synthetic POC gate, not a claim of generalization to real services or superiority
to the adaptive checklist. Abstentions count as unresolved, not correct.

## Diagnostic experiments

1. **Penalty-only (v2):** seed 0, 50,000 driver steps, original architecture,
   replay and training ratio. Change only training escalation/timeout penalty
   from 0.5 to 2.0. Evaluate with the original penalty of 0.5 and fresh seeds
   2000000 / 2100000. Preserve the original training evidence.
2. **Inspection coverage (v3), if necessary:** return to the original penalty.
   During training only, assign 50% of episodes a two-inspection sequence in
   random order followed by a uniformly random diagnosis. This exposes sensor
   outcomes without hidden labels, expert diagnosis or supervised imitation.
   The actual executed action is written both to replay and recurrent carry.
   Remaining episodes use the learned policy. Use 50,000 steps, training seed 0,
   and new evaluation seeds 3000000 / 3100000. Evaluation disables all guidance.
3. If a development condition meets the gate, freeze it and run confirmation
   seeds 10, 11, 12 with evaluation offsets 4000000 / 4100000. Do not tune the
   selected condition using confirmation results. Report every seed.

These two experiments isolate different candidate causes; they are not a full
factorial study. Changes beyond them must be logged as new development trials,
with fresh confirmation data. Do not claim causal certainty from one seed.

## Boundaries

Use only systems-world-model-lab, sequential jobs, the existing four-CPU / 6 GiB
quota and default-deny networking. No dependency downloads are needed. Keep
original source, checkpoints and evaluation data. No LLM or other namespace
participates in these synthetic experiments. Record runtime and failed trials.

## Development extension (before v4 evaluation)

v2 changed abstention into chance-level guessing without inspections. v3a
inspection coverage retained near-total abstention. A separate 90-episode probe
with forced inspections found 76.7% correct reward ranking by the world model,
but 0% correct policy diagnoses. This is not autonomous accuracy.

v4 combines coverage 0.5 and abstention penalty 2.0, and increases the budget to
100,000 driver steps, seed 0, fresh evaluation offsets 6000000 / 6100000. This
changes both training incentives and data coverage plus compute budget; do not
attribute an improvement to any one factor. The environment and original
evaluation scoring stay fixed. If successful, confirmation uses seeds 10-12
and offsets 4000000 / 4100000 as above.
