# Independent confirmation of the learned-state readout

Prespecified before model training or new incident collection. The previous model-22 result (80/80 latent, 77/80 primary reward) motivated this candidate selection; it is development evidence, not confirmation. The previous failed primary gate remains failed.

## Frozen candidate and comparison
- Train actual pinned upstream DreamerV3 from scratch with unused model seed 23, 100,000 steps, the existing 160 calibration incidents, and unchanged auxiliary supervision (weight 1). No resume or new training data.
- Fit the same three-label ridge classifier to the mean of 16 posterior particles (RSSM deterministic and stochastic state), after fixed CPU, pool and trace inspections. Fit normalization on calibration only; ridge penalty 1.0; signed-label threshold 0.0. No tuning or test-time refitting.
- This recipe is frozen, not the seed-22 coefficient matrix: an independently trained model has a different latent coordinate system and requires calibration-only fitting of its own coefficients.
- Primary outcome is now the latent-state classifier. Raw telemetry ridge, shuffled-label latent ridge, auxiliary logits and original imagined-reward argmax are secondary comparators. No secondary result can replace a failed primary outcome.

## Fresh evaluation and decision
- Scenario seed 6801, 80 freshly collected incidents, ten per known family: healthy, CPU, pool, delay, CPU+delay, CPU+pool, pool+delay, triple. Shifted load/severity as in development.
- Two awaited warm-up batches, then measurement-only reset; four measured frames. Check every service's cumulative counts n, 2n, 3n, 4n; all requests must succeed and the incident must complete within 35 seconds.
- Gate: healthy 10/10 and at least 9/10 exact fault sets in each other family. Report all errors. No relabeling, case removal, seed reroll or threshold changes based on outcome.
- Training data retains historical warm-up samples; the new model and probe use exactly the same data contract as the selected development candidate. Startup failures require a separate future scenario.
- This is independent model initialization and a fresh scenario batch, not an independently sourced training corpus or proof of unseen-system generalization. A pass supports a limited known-regime POC only; it does not establish autonomous remediation or production reliability.
- No automatic serving promotion. Restore all original sources and replicas after testing. Only systems-world-model-lab may be changed.

## Provenance
Source ConfigMap: microservice-source-v19, SHA256 e6a27c2ffda41ae56ffa905a037f1e644e8c14c6700518e6dd4540f39bd450eb.
Calibration SHA256: e6d217c0fcdddc2b50df296c09c95540eb3223ff5d7fcf6f0c9f179b6d903473.
Upstream revision: e01491fad6434b2245a3b8ca201dd7faedcc458c.
Fresh output: /results/micro-latent-confirm-v1/; prior artifacts remain untouched.
