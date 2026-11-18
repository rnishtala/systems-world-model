# Mechanism-specific diagnostic readouts

2026-11-18. Development experiment; model 81 remains frozen.

## Hypothesis

CPU, pool contention and delay may require different representations and regularization. The existing readouts use one representation and one hyperparameter configuration for all three outputs. Test whether selecting each output separately reduces combined-fault errors.

## Fixed procedure

Use only the original 128 training incidents (seed 8701) and saved model-81 features. Primary: independently select each mechanism's representation from load response, Dreamer latent state, and response plus forecast features, and its configuration from the existing eight candidates. Select on mechanism accuracy, with deterministic candidate-order ties. Compare against a matched telemetry-only selector using response and history, and the original shared-config latent and response readouts.

Use nested leave-one-workload-regime-out cross-validation: four outer groups, three inner groups. Normalization and fitting use the respective training fold only. Save every split, selection, prediction and source hash. No confirmation rows or labels participate in this experiment. Separately tabulate errors in the already-completed confirmation as descriptive evidence only.

The development gate requires at least 116/128 exact diagnoses, healthy 16/16, every other family at least 12/16, and at least four more correct than BOTH the matched telemetry selector and original latent readout. Do not tune or retry based on the result. A failed gate stops this readout experiment. A passing gate warrants a separately specified fresh confirmation, not promotion.

The world model trained on all 128 incidents, including supervised mechanism labels. These folds evaluate readout selection only: they are optimistic for world-model generalization and cannot establish superiority on unseen incidents. No Dreamer weights, serving model or cluster resources change.
