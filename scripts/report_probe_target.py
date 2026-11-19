"""Render the frozen feasibility comparison without promoting secondary outcomes."""
import json
from scripts.verify_probe_target import R,O

def main():
    s=json.loads((O/'evaluation/summary.json').read_text());m=s['methods'];meta=json.loads((O/'models/training-91.json').read_text())
    lines=['# Probe-target-aware Dreamer result','',
      '2026-11-18. [Protocol](PROBE_TARGET_MODEL_PROTOCOL.md) · [Evidence](evidence/probe-target-model-v1/evaluation/summary.json).','',
      'The prespecified feasibility gate '+('passed' if s['feasibility_gate'] else 'failed')+'. A new upstream DreamerV3 checkpoint (seed 91) was trained from scratch with probe target and concurrency as actions. The serving checkpoint was not changed.','',
      f"Training used 64 measured eight-observation trajectories and 100,000 environment steps, taking {meta['seconds']/60:.1f} minutes. These trajectories came from the earlier probe pilot, including its former test set; they are training data in this new experiment. The comparison below uses 32 newly collected incidents (seed 9202), after readouts and checkpoint were frozen.",'',
      '| Method | Exact diagnoses |','|---|---:|']
    for n,title in [('latent','Dreamer latent readout (primary)'),('response','Telemetry response + ledger probe'),('history','Full observation history'),('response_history','Response + history'),('auxiliary','Dreamer auxiliary head'),('shuffled_latent','Shuffled-label latent control')]:
        lines.append(f"| {title} | {m[n]['correct']}/32 |")
    lines+=['','| Criterion | Result |','|---|---|',
      '| Primary reliability thresholds | '+('Pass' if s['reliable'] else 'Fail')+' |',
      '| Within one correct diagnosis of strongest telemetry control | '+('Pass' if s['within_one_of_strongest_control'] else 'Fail')+' |',
      '| Ledger forecast MSE below persistence | '+('Pass' if s['forecast_gate'] else 'Fail')+' |','',
      'Reliability thresholds are >=29/32 overall, healthy 4/4 and each fault family >=3/4. They are screening criteria, not confidence bounds on production reliability. Passing feasibility would not establish diagnostic superiority.','',
      '| Family | Dreamer primary | Response control |','|---|---:|---:|']
    for f,title in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','All three']):
        lines.append(f"| {title} | {m['latent']['families'][str(f)]['correct']}/4 | {m['response']['families'][str(f)]['correct']}/4 |")
    lines+=['','## Measured-action forecasting','',
      '| Transition subset | Count | Dreamer MSE | Persistence MSE |','|---|---:|---:|---:|']
    for name,v in s['forecast_mse'].items():lines.append(f"| {name} | {v['n']} | {v['predicted']:.6f} | {v['persistence']:.6f} |")
    lines+=['','Forecasts predict next ledger latency in log1p(seconds/0.1) units. Decoder symlog is inverted per particle before averaging. Comparisons cover measured actions only; no unobserved counterfactual effect or autonomous probe selection was tested. Persistence is a naive forecast control; this does not demonstrate superiority over a learned forecasting baseline.','',
      '## Paired diagnosis comparisons','']
    for name,c in s['paired'].items():lines.append(f"- Against {name}: Dreamer-only correct {c['latent_only_correct']}, control-only correct {c['control_only_correct']}; two-sided exact McNemar p={c['p']:.4f}.")
    lines+=['','## Interpretation','']
    if s['feasibility_gate']:
        lines+=['The new model meets the small-sample feasibility criteria for measured probe transitions and diagnosis. The useful next question is whether model predictions reduce probe count or observation cost under a fixed budget, followed by a larger frozen evaluation. Equally informed telemetry controls remain essential; matching their accuracy alone is not evidence that a world model is necessary.']
    else:
        lines+=['The new contract was exercised end to end, but the model did not meet all feasibility criteria. Preserve the fixed comparison and identify which criterion failed before further training. Do not reinterpret a successful secondary readout or forecast subset as a pass for the primary.']
    lines+=['','## Verification and limits','',
      'The actual upstream smoke run checked normalized two-dimensional actions, rejection of unmeasured replay actions, checkpoint loading, posterior extraction, prior forecasts, label exclusion and the offline-only serving guard. Local verification reproduced training-only readout selection/fits, input observations, action alignment, predictions and metrics; checked source/export hashes, data provenance and checkpoint/readout immutability assertions. Latent states and prior forecasts were not regenerated locally.','',
      'Both target choices are encoded as endpoints of a bounded float action component; replay accepts only the measured endpoint and concurrency. Dreamer uses its recurrent world model and auxiliary mechanism supervision. Its actor is overridden by the measured acquisition schedule, and reward is zero: this is not a learned investigation policy. The encoded observation also includes the declared next action. Before testing alternative-action planning, separate that scheduling metadata from the encoded state and validate action effects on randomized measured transitions.','',
      'The experiment uses one model seed, 64 training and 32 test incidents, the same eight controlled performance-fault families and 3/9 concurrency values. All requests succeeded. Pilot-derived observations and protocol choices influenced this research direction. Development readout CV is not independent world-model evaluation because the model saw those training trajectories. Small sample size, fixed topology, sequential probes and previously studied fault families limit generalization claims. The older workload-only model used different observations and test incidents, so these scores are not a controlled before/after accuracy improvement.','',
      'All compute and fault injection stayed in systems-world-model-lab. Inference was restored to its exact prior template and replica count, experiment resources were removed, and the existing serving model was not promoted or replaced.','']
    (R/'reports/PROBE_TARGET_MODEL_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
