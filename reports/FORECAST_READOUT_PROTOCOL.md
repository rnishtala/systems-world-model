# Forecast-informed diagnosis development protocol

2026-11-18. Frozen before calculating candidate results.

Use only the original 128 training incidents (seed 8701), saved model-81 states and corrected forecasts. No validation incidents or labels enter this development program. The world model already trained on these incidents and mechanism labels, so readout cross-validation cannot establish independent world-model generalization.

Primary representation: 14 load-response features plus predicted change from current trace and observed-next-trace minus prediction residual, across three transitions and four latency channels (24 additional features). All methods receive four observations; residuals are available only after their target observation. No hidden counters, fault controls, labels or regime identifiers enter features.

Controls: load-response alone, history, latent state, load-response plus history, and load-response plus latent state. Use the same existing eight linear/RBF candidates for each. Outer cross-validation holds out each of the four workload regimes (32 incidents). Inner selection holds out each of the three remaining regimes, normalizing and fitting only inner training rows. Select by pooled inner exact diagnoses, first-candidate ties; refit on outer training rows before predicting the held-out regime.

Primary development gate: >=116/128 exact, healthy 16/16, every other family >=12/16, and at least four more correct than load-response alone. Failure means no new incidents or training. If it passes, select final configurations with four-regime CV on training rows, fit once on all 128, and freeze files before collecting fresh validation: seed 8902, 64 incidents, existing 3/7 and 5/9 regimes. Fresh validation requires >=58/64, healthy 8/8, other families >=6/8, and at least two more correct than load-response. No automatic final test or promotion.

Any live collection stays in systems-world-model-lab with existing leases and restoration safeguards. Preserve all original evidence and checkpoint files.
