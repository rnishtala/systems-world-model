# Ledger probe operational recovery

2026-11-18. Amendment before replacement test collection.

The seed-9102 test attempt was interrupted when repeated SSH monitoring failures raised out of the collection loop. The original runner deleted the job in its cleanup path before exporting test observations. No test rows or predictions were exported or scored. Development data and frozen readouts remain unchanged; inference was restored and verified ready.

Retain this failed attempt as operational evidence. Collect one replacement test with seed 9103, preserving all other collection, fitting, feature and success-criterion choices. This is an explicitly documented operational replacement, not a score-driven reroll. The collector source differs only in the test seed; retain both source hashes. The original no-reroll rule still applies to measurement failures and disappointing scores.

The recovery runner catches monitoring failures until a bounded overall deadline instead of immediately deleting a healthy job. It periodically saves raw collection logs locally, preserving partial evidence. It never restarts a job following a connection timeout. Readouts stay frozen, and the original development collection is not repeated. Deployment cleanup and exact inference restoration remain required.
