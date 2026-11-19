# Training dataset uploads

Upload UTF-8 JSONL: one incident object per line, 64–256 incidents, at most 2 MiB.
Use **Download selected dataset** in the experiment form to inspect a real example.
Uploads are immutable, hashed, stored on the lab results volume, and selectable only
by their uploader and the administrator. A successful upload does not start training.

## Incident fields

- `seed`: collection seed, integer 1–999,999,999. This is not the model seed.
- `episode`: nonnegative incident index, unique within its collection seed.
- `fault`: integer label 0–7, using the fault mapping below.
- `repeat`: nonnegative repeat index; `repeat % 4` defines a training-only validation group.
- `regime`: 0 (original operating conditions) or 1 (shifted conditions); optional for legacy training rows.
- `split`: optional legacy marker; only `train`, `training`, or `development` are accepted.
- `utc`: real collection timestamp in Unix seconds, not a future timestamp.
- `seconds`: actual collection duration in seconds, at most 35.
- `base`: four measurement windows, all targeting `checkout`.
- `order`: `["checkout", "ledger"]` or `["ledger", "checkout"]`.
- `extra`: an object with `checkout` and `ledger`, each containing two windows.

Include every fault family at least eight times, with all four `repeat % 4` groups
represented within each family. Duplicate incident IDs and duplicate trajectories
are rejected. Do not relabel or duplicate test incidents to satisfy these checks.

| Fault | Injected mechanisms |
| --- | --- |
| 0 | Healthy |
| 1 | Payment CPU |
| 2 | Ledger pool contention |
| 3 | Ledger delay |
| 4 | CPU + delay |
| 5 | CPU + pool |
| 6 | Pool + delay |
| 7 | CPU + pool + delay |

## Each measurement window

A window contains exactly `target`, `requests`, `successes`, `metrics`, and `seconds`.
`requests` is an integer 1–12; `successes` must equal it. `metrics` contains `checkout`,
`payment`, and `ledger`. Each service contains:

- `count`: integer; equal to requests for all services when targeting checkout,
  or only ledger when targeting ledger (other service counts must be zero).
- `latency_p95`, `self_p95`, `downstream_p95`: finite, nonnegative seconds (at most 35).
- `error_rate`: zero for this measured-probe protocol.

The current model learns latency fault diagnosis from eight successful measurement
windows. Arbitrary logs, failed-request traces, CSVs, other service names, and raw
RCAEval datasets need a separate adapter; this uploader does not convert them.

## Keeping evaluation separate

Confirm the file contains training incidents only. Known historical held-out seeds
and exact measurement trajectories from retained launcher test runs are rejected.
Fresh evaluation uses separate seeds and is collected after readouts are frozen.
Checks cannot establish the provenance or correct labels of an arbitrary external
file: the uploader remains responsible for accurate training-only provenance.

Training seeds are below one billion; launcher evaluation seeds are at or above
one billion. Limits are five uploads per user and twenty across this lab, with at
least 1 GiB free storage required. No checkpoints or uploads are committed to Git.
