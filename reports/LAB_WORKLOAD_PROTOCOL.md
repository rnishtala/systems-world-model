# Workload-conditioned trace prediction protocol

Frozen before the lab-workload-v1 run, 2026-11-18.

## Question and scope

Does explicitly conditioning DreamerV3 transitions on workload improve trace forecasting and diagnostic representation? This is empirical trajectory replay, not a simulator capable of answering arbitrary counterfactuals or an autonomous remediation policy.

Reuse the exact 128 training incidents from lab-regime-v1 (seed 8701, SHA256 e83466b6e689e0e81d7575dc4b4b4e824eec4360daa3f250e81e332f9d51b7eb). No previously collected validation incidents enter training. Train a fresh seed 81 for 100,000 steps, same size1m network configuration and auxiliary mechanism-label loss. The action head changes from discrete diagnoses to a continuous workload scalar; this also changes the replay policy and episode termination. This is not a single-variable causal ablation against model 71.

## Action and observation contract

A workload action is the next batch's concurrency divided by 12 in the raw environment. Dreamer's upstream wrapper maps this to [-1,1], so the agent carry and replay use 2*n/12-1. Each recorded transition must match the workload actually measured; reject mismatched actions instead of fabricating a counterfactual observation. The replay policy always follows the measured schedule.

The 14-dimensional trace-log view is retained, with unused index 6 now carrying the declared next workload (known by the experiment scheduler before the batch). No future latency, hidden resource counters, fault labels, seed or regime IDs enter observations. Four observations are followed by a zero-workload terminal bookkeeping step with duplicated final telemetry. Rewards are zero; the auxiliary target is optimized only at the final nonterminal observation. This does not train a useful actor or establish diagnostic reward optimization.

Before each next observation, forecast its four log-latency channels from the current posterior and scheduled workload. Score only the three real transitions, never the bookkeeping transition. Compare mean squared error across all incidents, transitions and channels to persistence (the current latency). All posterior inference uses zero mechanism targets; verify label invariance on the actual checkpoint.

## Preconditions

Run synthetic trajectories through the pinned upstream action wrappers, verify action normalization, reject unmeasured actions and check no future telemetry enters the current observation. Train a separate 256-step smoke checkpoint, load it, run label-isolation checks, and exercise posterior extraction and prior prediction. Only then train the full checkpoint. Smoke output is separate and never resumed.

## Evaluation and gates

After checkpoint freeze, collect 64 fresh validation incidents with seed 8802: eight fault families, two workload regimes (3/7 and 5/9), four incidents per family/regime with balanced forward/reverse order. Use existing fault severities, bounded leases, independent measurement windows and failure-retention rules from LAB_REGIME_PROTOCOL.md.

Compare state readout, full history, latest snapshot, and the frozen load-response feature design in microservice/load_response.py. Each uses the same four observations; history/latest also receive the announced schedule. Fit the same eight linear/RBF candidates on training data only; validation selects the existing fit without refitting. Include auxiliary-head and shuffled-training-label controls (permutation seed 8805).

Screening requires state diagnosis >=58/64, healthy 8/8, every other family >=6/8, no more than two correct behind the strongest of all three telemetry baselines, AND forecast MSE strictly below persistence. Validation is selection/development evidence.

If screening passes, freeze readouts and collect 80 fresh final incidents, seed 8803, workload 3/11/6/9 or reverse, ten per family. Final diagnosis gate: healthy 10/10, every other family >=9/10, and >=4 more correct than the strongest telemetry baseline. Report final forecasting separately; a claim of useful workload modeling also requires lower final forecast MSE than persistence. If screening fails, collect no final-test data. No model promotion in either outcome.

## Isolation and recovery

Only systems-world-model-lab is modified. Retain established security and 2-CPU/4-GiB job limits, pause inference within quota, restore its exact template and replica count, and export evidence read-only with hashes. Do not modify other namespaces or AutoSRE. Preserve failed-run evidence; do not silently retry training or discard incidents.
