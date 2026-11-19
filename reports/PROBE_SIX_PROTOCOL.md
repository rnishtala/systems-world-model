# Fresh confirmation of a fixed six-window diagnosis budget

Preregistered before fresh collection. Motivation: the post-hoc six-window Dreamer readout scored64/64 in the previous optimistic training-data development screen. This experiment tests that candidate on new incidents; no further prefix or threshold search.

Freeze model91 unchanged, plus supervised readouts fitted exclusively on64 training incidents (seeds9101/9103). Select readout configurations by leave-repeat-out CV on training data from the same existing eight-candidate grid. Methods: latent6(primary), latent8, history6/8, summary_history6/8, and the prior response8 telemetry representation. Freeze all parameters/configs and training/model hashes before collection. No previous test9202 rows used for fitting.

Collect32 new incidents using seed9302, four per eight fault families, the unchanged loads3/9 and balanced probe orders. Preserve all eight measured windows for paired shadow scoring. Six-window predictions receive only the first six measurements, with planned action metadata as in training; no test labels or future measurements enter inference. No new training or online model promotion.

Primary feasibility gate: latent6 >=29/32 exact diagnoses, all four healthy correct, >=3/4 for every family, at most one fewer correct than latent8, and at most one fewer correct than the strongest full-eight telemetry control. Report exact paired wins/losses and two-sided binomial discordance p-values against all controls. Report six-window controls explicitly: a tie means no model-specific advantage. This small fixed-sample gate is engineering feasibility, not statistical proof of noninferiority or production reliability.

Expected shadow cost: six vs eight windows,36 vs48 ingress requests (25% saving). Warmup36 requests are excluded; including warmup,72 vs84 total requests is only14.3% saving. The experiment itself still collects all eight windows. Do not claim actual runtime savings or adaptive action selection. Do not reroll invalid incidents or tune after results. Preserve any failed run and state failure.

Scope systems-world-model-lab only. One bounded job with existing resource limits, no service account token; temporary inference pause and exact restoration. Existing45-second fault leases,35-second incident bound, reset after each incident and in finally. Preserve evidence, verify checkpoint/readout immutability, delete temporary resources. No other namespace or AutoSRE changes.
