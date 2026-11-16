# RCAEval diagnosis-supervised development experiment

Scope: systems-world-model-lab only; fixed 73 training and 25 validation
incidents. Original test data/predictions remain frozen and excluded from bundles.

Train fresh seeds42 and43 for32,000 steps each, matching the preceding
unsupervised experiment. Retain original upstream dynamics/reconstruction losses,
normalization, architecture and zero-reward no-op environment. Add a linear
root-cause service head with cross-entropy weight1 on post-onset states only.
Training service labels are loss targets, never encoder/decoder inputs. All
policy calls strip the target field; validation inference receives dummy zeros.
An actual model audit must prove encoder/decoder exclusion, optimizer inclusion,
and identical policy outputs under opposite supplied targets at a fixed RNG key.

Primary metric: top-1 service accuracy for the existing post_mean_full_ridge100
readout (16 posterior particles), separately for each seed. Comparators are the
preceding unsupervised seeds16/25 and19/25, and raw telemetry23/25. Retain the
existing readout menu, shuffled-label and missingness controls as secondary
analyses. No selecting a seed or touching the test set. Validation gains are
model-development evidence, not independent confirmation or grounds to promote.

Fixed resource cap2 CPU/4Gi; unchanged namespace quota and network default deny.
Temporarily pause only lab inference to fit quota, then restore exact template
and replicas. No changes to the serving package or other namespaces. Maximum
combined job duration30 minutes; no automatic retries or budget extensions.
