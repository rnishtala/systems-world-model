# Active diagnostic observation experiment

Frozen before downloading RE2-TT telemetry or scoring any held-out predictions.

## Question

Can action-conditioned Dreamer predictions choose which service to inspect next, reaching the correct diagnosis with fewer observations than simpler acquisition strategies?

Use fresh RCAEval RE2-TT (Train Ticket), dataset revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`. Repetitions 1–2 train and 3 test; expected 60/30 incidents. Apply the existing published-onset timestamp validity rule before inference. Root services in training define five candidate services/actions. Train-derived metric vocabulary is restricted to those candidates. This is a closed-set experiment, not unknown-service detection.

## Observation task

Each incident is a static five-minute post-onset telemetry window. Existing preprocessing normalizes per metric using the preceding five minutes, bins into 20 frames, and includes observation masks. Average the ten post-onset frames to form a vector of metric values and mask fractions. Initially all post-onset telemetry is hidden. A check reveals one candidate service's aggregate telemetry; previously revealed values persist. Service-seen bits distinguish unqueried data from absent metrics. Five checks reveal every candidate service. A check is an offline reveal, not a Kubernetes operation or an actual remediation intervention. No simulated future readings are treated as observed evidence.

Each policy generates a sequence of five distinct checks. Score diagnosis after each budget 1–5. No adaptive stopping or test-calibrated confidence threshold is claimed. All methods use one identical diagnosis classifier fitted from training incidents under all 32 service subsets: standardized ridge regression, penalty 100. This holds the diagnosis mechanism constant and compares only acquisition.

## Methods frozen before evaluation

1. Dreamer lookahead: train actual upstream DreamerV3 size1m from scratch with action-conditioned random inspection trajectories, 64,000 environment steps, seeds 42/43, original world-model/actor objectives plus weight 1 training-only service cross entropy. Reward is always zero; the native actor is not the acquisition controller. Override training actions with uniformly randomized full service permutations and update recurrent previous-action state to the executed action. At inference, imagine each available inspection using 32 latent samples, apply the learned diagnosis head, and choose the lowest mean predicted categorical entropy. Never call the observation oracle while comparing candidate checks.
2. Fixed order: alphabetical service order, declared in advance.
3. Random order: 16 fixed permutations per incident, seeds derived from fixed seed 1901, incident ID and replicate. Aggregate over the same 30 incidents, not 480 independent incidents.
4. Retrieval-adaptive: use only training-case telemetry and labels. Normalize distance dimensions by training standard deviation. Weight training examples by softmax of minus mean squared distance on already-revealed metrics. For each unseen service, use training examples as possible readings and choose minimum expected label entropy after incorporating that service's distance, weighted by current example probabilities. Temperature 1, stable tie break. No test labels or unseen actual values enter this decision.

Both seeds use the same fixed training split and diagnosis classifier. No selection among seeds or controllers. Report all curves, paired disagreements, check orders and elapsed times.

## Decision rule

Primary measure is mean service-localization accuracy over budgets 1–4, equally weighted. A useful signal requires Dreamer to exceed fixed and retrieval-adaptive acquisition by at least 5 percentage points in each seed; random is an additional reference. Also report whether budget 3 gets within one correct incident of all-five-check accuracy, and the smallest fixed budget meeting that criterion for each method. Treat that smallest budget as a descriptive metric, not a tuned deployment policy. If full-information classifier accuracy is below 29/30, flag the diagnosis ceiling and do not claim reliable triage regardless of acquisition score. These small-sample criteria are exploratory, not statistical proof.

All predictions, action sequences, classifier hashes and checkpoints are frozen before a separate local scorer opens held-out labels. Training-only labels are excluded from policy/encoder/decoder observations; run actual model label-isolation audits. Unit checks must enforce that an unqueried service cannot affect a policy decision and that revealed values match the queried service only. Preserve negative results; no test-guided retraining.

Only systems-world-model-lab; bounded no-egress job under existing quota; pause and restore only the inference deployment. Use a fresh output directory, archive evidence, remove experiment pods, retain checkpoints and the 24-hour Job TTL fallback. No serving promotion.
