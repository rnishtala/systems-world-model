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

## Initial boundaries (concurrency amendment recorded below)

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

## Separate model-lookahead controller (prospective development evaluation)

If the native actor remains ineffective, evaluate a frozen Dreamer world model
with a separate two-step Monte Carlo lookahead controller: 16 first-step samples
and four continuation samples per diagnosis. Estimate immediate outcomes for all
seven actions; for information actions, add the best expected subsequent
diagnosis reward, using the learned continuation prediction. No simulator calls
or hidden labels are available to the planner.

Each inspection and waiting are capped at once by explicit controller rules.
This guarantees at most two inspections by design; it is not evidence of learned
inspection efficiency. This controller is a harness addition, not Dreamer's native
actor. Its results must be reported separately. Development evaluation uses
150 episodes per regime and offsets 7000000 / 7100000; all baseline scoring stays
unchanged. The learned transition and reward model, not expert fault thresholds,
supplies the action scores. Planning quality and real transfer remain unproven.

## Native actor argmax evaluation (prospective development comparison)

The combined native sampled actor achieved 93.3% / 94.7%; two-step model
lookahead achieved 94% / 94%. Before further training, evaluate the same native
actor with categorical argmax action selection, leaving the RSSM stochastic and
adding no action restrictions. Use fresh offsets 8000000 / 8100000, 150 episodes
per regime. This is an inference choice, not retraining or an expert diagnosis
rule. Preserve sampled-policy results and identify deterministic selection
explicitly. Confirmation, if selected, must use this same controller throughout.

## Native latent-particle vote (prospective development comparison)

Argmax action selection did not improve the single-particle actor: 93.3% / 94%.
Test 16 native RSSM/actor particles with identical real observations; choose the
most-voted action and feed that actual action back to every particle's recurrent
state. This changes inference only, adds no inspections or expert fault rules,
and leaves all seven actions available. Use offsets 9000000 / 9100000 and 150
episodes per regime. Retain all prior development results. This is the last
planned inference-only comparison before reassessing the remaining learning gap.

## Frozen confirmation configuration

The 16-particle actor achieved 100% / 100% on development episodes, with one
inspection per episode. Freeze training at 100,000 steps, penalty 2.0, coverage
0.5, and particle voting at 16. Run seeds 10, 11, 12 from scratch with offsets
4000000 / 4100000. Both native sampled and particle-vote results are preserved.

To reduce wall time without increasing the namespace's four-CPU / 6 GiB caps,
run three confirmation jobs concurrently with limits of 1333m CPU and 2 GiB
memory each. Raise only the namespace's pod-count quota from two to three.
Use one intra-op thread per job and separate source directories extracted from
the same frozen bundle (Python bytecode writes disabled); each seed has separate
logs, replay and checkpoints. This
changes execution concurrency, not the frozen model or learning configuration.
Do not tune against confirmation results or silently replace failed seeds.

## Recorded confirmation outcome (after the frozen evaluation)

All three seeds met every threshold: 100% familiar and shifted accuracy, one
inspection per incident, zero unresolved cases, 150 episodes per regime per seed.
The batch completed in 26.7 minutes. The unchanged native sampled actor reached
90.0% / 93.8% aggregate accuracy; the confirmed result specifically includes
16-particle inference. The separate saved-checkpoint demo diagnosed all three
fault classes correctly. See `evidence/followup/confirmation-particle-gate.json`
and `evidence/followup/confirmation/` for the complete episode records.

This closes the synthetic POC gate, not real-service transfer or LLM resolution.
