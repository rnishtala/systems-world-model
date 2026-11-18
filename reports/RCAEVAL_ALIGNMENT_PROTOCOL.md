# Development check-score alignment experiment

Freeze before execution. Reuse frozen acquisition models 42/43, only the same
60 training incidents; never read the 30 held-out incident arrays or labels.
This is development evidence, not an independent validation of Dreamer: both
world models have already seen these incidents.

Partition each root-service class into two equal folds by sorted incident ID.
Fit the shared ridge classifier on the opposite fold (all 32 observation masks,
penalty 100). Fit clipping bounds and feature scales on that opposite fold too.
This removes diagnostic-classifier self-fitting, but does not undo world-model
training exposure. No model retraining, threshold search, or promotion.

Compare four full five-check trajectories: original auxiliary entropy; mean
entropy of the cross-fitted ridge classifier on each imagined particle; the same
with predicted query values clipped to opposite-fold min/max and masks to [0,1];
and fixed alphabetical order. Preserve all observed coordinates and revealed bits;
change only the proposed service's predicted readings and set its seen bit.
Unqueried coordinates remain zero. Rank by mean per-particle categorical entropy,
using 32 particles and stable alphabetical tie breaks. All methods use the same
opposite-fold ridge classifier for actual diagnosis. Report curves for budgets
1–5 and primary development mean accuracy across budgets 1–4, both seeds, all
methods. No seed or controller selection from the held-out test set.

Separately audit decoder errors on the fixed-order prefixes for every available
candidate: per-feature raw MSE, variance-standardized MSE, standard deviation,
mean baseline MSE, and range-violation rate; report metric values and masks
separately. The audit uses full training statistics solely for descriptive
comparison with the previous audit. No candidate observations enter prediction.

Run only in systems-world-model-lab with the existing bounded, no-egress Job,
namespace quota and 24h TTL. Pause/restore only inference. Check checkpoints
against original hashes, archive outputs and logs, remove experiment resources.
