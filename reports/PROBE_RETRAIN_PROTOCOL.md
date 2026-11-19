# Expanded-data Dreamer weight training

Registered before training or collecting test data. Hypothesis: training the recurrent world model on broader conditions reduces false pool diagnoses beyond refitting a readout alone.

Reuse exactly 128 designated training incidents: seeds 9101/9103 (64) plus 9501 (64). Exclude all previous tests, including 9502. Train upstream Dreamer from scratch, seed 101, 100,000 steps, size1m, existing trace-probe-target-v1 contract and mechanism supervision. This changes world-model weights; it is not just readout fitting. One training seed cannot isolate dataset effects from initialization variance.

Fit latent6 (primary), latent8, history6 and history8 readouts using the same eight candidates and leave-repeat-modulo-four-out selection on training data only. Freeze weights and readouts before fresh test seed 9503: 64 incidents, eight families x four repeats x two regimes, same original/shifted collector settings as the preceding experiment. All eight windows are collected for shadow prefix comparison; no new actual latency-savings claim. Labels never enter encoder inputs. No rerolls or test-driven tuning.

Paired comparators on identical fresh incidents: old Dreamer91 with original latent6 readout, old Dreamer91 with expanded-data latent6 readout (primary comparator), and telemetry histories trained on the same 128 rows. Primary gate: new latent6 >=58/64, healthy8/8, each family >=6/8; each regime >=29/32, healthy4/4 and each family >=3/4; >=2 more correct than expanded-readout Dreamer91 and strictly fewer false pool predictions; within two correct of history6 and new latent8. Report exact paired discordance p-values as exploratory, not multiple-comparison-adjusted proof.

Scope only systems-world-model-lab. One job bounded 3600 seconds, 2CPU/4GiB, no service-account token. Separate processes release training/inference memory. Existing fault leases45s, incident cap35s, per-incident/final reset. Pause serving inference temporarily and restore exact spec/replicas; no promotion, no other namespace changes. Export evidence with hash verification and remove temporary resources.
