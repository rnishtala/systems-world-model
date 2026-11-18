# Fresh confirmation of training-only frozen readouts

2026-11-18. This is a new experiment following the failed forecast-feature development gate. It does not relabel that failure as success.

Primary: the model-81 latent-state readout, selected using only the 128 original training incidents. Controls: load response, history, load response plus history, load response plus latent, and load response plus forecast changes/residuals. The prior development results motivate this new primary; only entirely fresh incidents can now support confirmation.

Select each readout's configuration using leave-one-workload-regime-out CV across the original four training regimes with the existing eight candidates, first-candidate ties. Fit the selected configuration once on all 128 training rows. Freeze parameters, configurations, source hashes, checkpoint identity and timestamp before collecting any new data. No validation-time selection, refit or model training.

Use the unchanged checkpoint 81, with the verified decoder-unit correction. Collect 64 fresh incidents with seed 9002, the existing 3/7 and 5/9 workload profiles, all eight fault families and balanced forward/reverse phases. Existing controls, severities, successful-batch requirements and fault leases apply. Reject and preserve a failed collection rather than discard incidents or reroll.

Report exact diagnosis and per-family scores for all readouts, auxiliary head, and corrected forecast MSE versus persistence. Primary reliability requires >=58/64, healthy 8/8 and every other family >=6/8. Added diagnostic value additionally requires >=4 more correct than the strongest of load response, history and load-response-plus-history. Report paired exact McNemar comparisons; effect sizes alone do not establish statistical significance. Forecast criterion is MSE below persistence. Report each criterion separately and their conjunction.

No final test beyond this confirmation, automatic promotion, or further collection. These incidents must stay outside both world-model training and readout fitting. The known fault families remain the same; this does not test unseen fault classes. Only the systems-world-model repository and namespace may change; restore the inference deployment exactly.
