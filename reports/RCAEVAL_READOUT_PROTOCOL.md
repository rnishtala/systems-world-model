# RCAEval validation-only readout study

All work stays in systems-world-model-lab. Original test predictions and scores
remain frozen. Use only 73 training and 25 validation cases; physically omit
all test arrays and labels from the input bundle. Existing model weights are
frozen. The reader loads the original 8,000-step checkpoint.

Before scoring, fix this menu: final deterministic state, post-onset mean
deterministic state, post-onset mean full state, raw sequence, and missingness-only
control. Average 16 posterior particles. Ridge penalties 1 and 100. Include
shuffled training-label controls; no shuffled label reaches observation inputs.
Select on validation only; this is model development, not new test evidence.
If no Dreamer representation matches raw validation accuracy, investigate longer
training with the same menu and no test access. Serving checkpoint never changes.

## Prespecified longer-training follow-up

The frozen-state study peaked at 14/25 validation diagnoses versus raw23/25.
Before running further fits, fix two fresh models with seeds42 and43, each
32,000 steps (four times the pilot), unchanged architecture/normalization/losses.
Apply exactly the same readout menu, ridge penalties, and shuffled controls.
Report each seed; do not select seeds for a favorable result. The no-op action
space remains unchanged for comparability; its exploration diagnostic is not
an outcome measure. Hold test arrays and labels out of the bundle entirely.
Resource limits remain 2 CPU/4Gi and the combined job deadline is 30 minutes.

The primary longer-run readout is `post_mean_full_ridge100`, the short-run
validation winner (14/25). This selection is recorded while the first32k model
is training, before either longer-run validation result is available. The other
menu entries are secondary diagnostics, not opportunities to select a seed.
