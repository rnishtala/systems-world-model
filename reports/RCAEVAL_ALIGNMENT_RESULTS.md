# Development score alignment — 17 November 2026

**Mixed result: matched scoring helps one frozen Dreamer seed, not both. No
promotion or independent confirmation.**

We compared four acquisition controllers on the same 60 development incidents,
using two equal service-stratified folds. Each diagnostic ridge classifier and
clipping bounds were fitted on the opposite fold. Both frozen Dreamer models
had already trained on all 60 incidents: this is only a readout cross-fit, not
world-model validation. The original held-out incident arrays and labels were
not loaded. All 2,400 model/controller/budget predictions are retained.

## Diagnosis accuracy by check budget

Entries are correct diagnoses out of 60. Mean is percentage accuracy averaged
over budgets 1–4; each incident contributes at every budget, not independent cases.

| Model / scoring | 1 | 2 | 3 | 4 | 5 | Mean 1–4 |
|---|---:|---:|---:|---:|---:|---:|
| 42: original auxiliary entropy | 26 | 33 | 42 | 47 | 48 | 61.7% |
| 42: matched classifier entropy | 23 | 31 | 45 | 47 | 48 | 60.8% |
| 42: matched, clipped forecasts | 25 | 30 | 44 | 45 | 48 | 60.0% |
| 43: original auxiliary entropy | 23 | 33 | 36 | 45 | 48 | 57.1% |
| 43: matched classifier entropy | 29 | 36 | 45 | 48 | 48 | 65.8% |
| 43: matched, clipped forecasts | 24 | 30 | 41 | 45 | 48 | 58.3% |
| Fixed alphabetical, either model | 24 | 32 | 36 | 42 | 48 | 55.8% |

Matched scoring changes seed 42 by −0.8 percentage points and seed 43 by +8.8
points relative to the original score. Clipping does not improve on matched
scoring in either seed. All variants exceed fixed order on this development
sample, but neither the two seeds nor the prior held-out negative result justify
a consistent benefit claim. The common classifier reaches only 48/60 (80%) with
all observations; check selection cannot fix that full-information ceiling.
Different diagnostic training sizes and cases mean this is not a regression
comparison against the previous 30-case test result.

## Why the earlier forecast error looked enormous

`ts-travel-service_error` accounts for **99.92% / 99.94%** of variance-standardized
measurement error. Its training standard deviation is only **0.0000727**, while
raw MSE is **0.000134 / 0.000182**. Dividing by that tiny variance inflates the
standardized MSE to about **25,393 / 34,428**. Predictions fall outside its narrow
observed range on **99.0% / 99.3%** of audited candidates.

This is a real relative forecasting error on a nearly constant feature, not proof
that the raw errors are physically large. Excluding that coordinate descriptively
reduces the remaining weighted standardized MSE to **0.866 / 0.871**. That exclusion
is diagnostic reporting only, not a tuned evaluation metric or deployed change.
The earlier aggregate alone was therefore insufficient to characterize overall
forecast quality. Clipping all query values to opposite-fold ranges did not
consistently improve acquisition; this experiment does not establish that the
single coordinate causes the diagnostic errors.

## Decision and next step

Keep serving unchanged. Before another external confirmation, build a genuinely
separate development split for world-model training and selection. Prioritize the
full-information classifier ceiling, using robust feature scaling/variance floors
and an explicit diagnostic baseline. Then test the matched acquisition score
across independent training seeds. Preserve the original held-out cases and use
a fresh untouched collection only after a repeatable development improvement.

The scoring change is an inference controller experiment around frozen DreamerV3,
not an LLM fine-tune or a newly trained world model. No causal remediation or
production-reliability claim follows from these results. Posterior-state sampling
adds variability and this small experiment does not establish statistical superiority.

## Verification and isolation

Nine unit tests passed, including preserved observed coordinates, hidden-value
isolation and clipping scope. Every method has exactly 60 predictions at each
budget with distinct checks. Fold IDs are disjoint. Exported hashes match;
checkpoint hashes match the original acquisition run. The Job took 48 seconds,
with about 22 seconds of evaluation per seed. Staging and restoration add wall time.

Only systems-world-model-lab was used. The inference deployment's prior template
and replica count were restored; the experiment Job and reader Pod were deleted.
Public UI, application services and serving weights were unchanged.

[Protocol](RCAEVAL_ALIGNMENT_PROTOCOL.md) ·
[Aggregate](evidence/rcaeval-alignment-v1/aggregate.json) ·
[Restoration](evidence/rcaeval-alignment-v1/restored.json)
