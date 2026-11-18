# Development check-order comparison

**Development gate passed; fresh confirmation is required.**

Nested selection on scenario8201 development data reached 80/80 diagnoses with 2.425 checks for Dreamer, versus 78/80 with 2.975 for telemetry. This meets the preset 0.5-check saving threshold against telemetry and all-three checking. These are exploratory development results after repeated analysis of this dataset.

## Per-order development results

These descriptive results do not choose the final policy. Each outer fold selected its order using its inner training results only. C=CPU, P=pool, T=trace.

| Order | Dreamer: two checks | Telemetry: two checks | Dreamer stopping: correct / mean checks | Telemetry stopping: correct / mean checks |
|---|---:|---:|---:|---:|
| C → P → T | 62/80 | 55/80 | 80/80 / 2.675 | 80/80 / 3.000 |
| C → T → P | 65/80 | 52/80 | 79/80 / 2.800 | 76/80 / 3.000 |
| P → C → T | 57/80 | 57/80 | 80/80 / 2.725 | 77/80 / 3.000 |
| P → T → C | 79/80 | 79/80 | 80/80 / 2.400 | 78/80 / 2.975 |
| T → C → P | 69/80 | 53/80 | 79/80 / 3.000 | 77/80 / 3.000 |
| T → P → C | 79/80 | 77/80 | 79/80 / 2.725 | 77/80 / 3.000 |

## Interpretation

Inspecting pool and trace before CPU substantially improves two-check diagnosis in this sample. Shared-prefix replay means the same observation history produces exactly the same saved latent state regardless of the eventual order. Actual previous-action inputs were verified 1,200 times. This is selection among fixed orders using supervised readouts, not per-incident adaptive planning or evidence of causal benefit from world-model objectives.

Outer-fold chosen orders vary: Dreamer 213/321/312/231/123; telemetry 231/231/123/312/123. This variability is a reason to require fresh confirmation rather than promote the best-looking development order.

## Operations and verification

Offline frozen-model replay ran only in systems-world-model-lab under deny-all network egress, without fault injection or training. Inference was restored unchanged; application services and public UI remained running. Temporary resources were removed. Ten targeted tests and full reproduction of nested choices, predictions, source hashes, shared-prefix identity and restoration passed.

[Frozen protocol](LAB_CHECK_ORDER_PROTOCOL.md) · [Verification](evidence/lab-order-v1/verification.json) · [Fresh confirmation protocol](LAB_ORDER_CONFIRM_PROTOCOL.md)
