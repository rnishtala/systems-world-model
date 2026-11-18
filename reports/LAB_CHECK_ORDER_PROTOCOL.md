# Development check-order comparison

Frozen before extracting new features. Use only the 80 saved scenario-8201
DEVELOPMENT incidents. Do not load scenario 8102/8202 evaluation data. Compare all
six permutations of CPU(1), pool(2), trace(3), in lexicographic order. Checks reveal
only their own group at the corresponding recorded frame. Replay assumes passive
inspection does not change system dynamics; it is not a live intervention test.
Both methods receive identical observation prefixes and unit check costs.

Extract features with frozen Dreamer seed 51 using a temporary offline Job confined
to systems-world-model-lab, with no service-account token and no fault controls.
Replay a prefix tree per incident: shared order prefixes reuse exactly the same
latent state, so stochastic sampling cannot advantage one order at a common
prefix. Before each child observation, write the actual inspection action into
Dreamer's carry. Labels never enter policy inputs; mechanism targets are zero.
Use the existing 16-particle averaged deter+stoch representation. Raw features
are flattened observed-prefix histories. Audit actions and label isolation;
verify checkpoint and source hashes before/after extraction. No retraining.
Only inference may pause for quota, then restore its exact template and replicas.
Application services and public UI remain running; remove temporary resources.

Use the same outer five folds (8301) and inner four folds (8302 + outer fold) as
the partial-readout study. All data-dependent choices stay within outer-training
cases. For each order, select each prefix's readout and calibrate stopping with
the unchanged partial_readout candidate grid and rules: linear penalties
1000/100/10/1, RBF gamma .1/1 crossed with penalty .01/.1, train-only normalization,
exact inner out-of-fold diagnosis accuracy for readout choice; stopping margins
0/.25/.5/1/2/4/infinity, requiring all healthy correct, >=90% per other family
rounded up, and <=1 fewer correct diagnosis than full checking.

Choose an order using inner results only: among calibrated candidates, minimize
mean checks, then maximize exact accuracy, then prefer lexicographic order. If
none calibrate, choose highest inner full-check accuracy (lexicographic tie), and
force three checks. This selects a fixed order per outer fold, not an adaptive
per-incident planner. Raw and Dreamer get the same candidate orders and readouts.
Score each outer case once using its selected order, readouts and stopping rule.
Also report per-order fixed-budget and stopping results descriptively; never pick
the winner based on outer scores. Inner model/order selection can overfit; only
outer predictions count for the development gate. Repeated use of this development
set means results are exploratory, not a fresh external validation.

Gate for a new live confirmation: selected Dreamer stopping gets healthy10/10 and
every other family>=9/10, loses<=1 case against its selected-order full-check
predictor, and saves>=.5 average checks versus BOTH three checks and the matched
selected-order telemetry policy. If this fails, stop and report; do not inject new
faults or promote a model. If it passes, freeze an all-development fitted policy
and a separate fresh confirmation protocol before collecting new incidents.

Record source/checkpoint hashes, extracted features, splits, all inner candidates,
selected orders, outer predictions and verification. This is a supervised readout
of a learned representation plus development-selected inspection order. It does
not establish native Dreamer planning or causal benefit of world-model training.
