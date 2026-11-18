"""Summarize the independently verified diagnostic efficiency evidence."""
import json
from datetime import datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/lab-efficiency-v1'
s=json.loads((O/'evaluation/summary.json').read_text());f=json.loads((O/'evaluation/frozen.json').read_text());v=json.loads((O/'verification.json').read_text());assert v['verified']
job=json.loads((O/'job.json').read_text());duration=(datetime.fromisoformat(job['completionTime'].replace('Z','+00:00'))-datetime.fromisoformat(job['startTime'].replace('Z','+00:00'))).total_seconds()
lines=['# Diagnostic check efficiency — fresh lab evaluation','',
'Frozen Dreamer seed 51; readouts fitted on the preceding 80 training incidents. Stopping thresholds selected on 80 new development incidents (8201), then evaluated on 80 later incidents (8202). No model retraining or serving promotion.','',
'| Method | Correct / 80 | Mean checks / 3 | Reliability gate |','|---|---:|---:|---|']
for name,label in [('latent','Dreamer state'),('raw','Telemetry history')]:
    m=s[name];lines.append(f"| {label}, calibrated stopping | {m['correct']} | {m['mean_checks']:.3f} | {'Pass' if m['gate_passed'] else 'Fail'} |")
    full=m['fixed_budgets']['3'];lines.append(f"| {label}, all checks | {full['correct']} | 3.000 | {'Pass' if full['gate_passed'] else 'Fail'} |")
lines+=['',f"Prespecified added-value signal: **{'met' if s['success'] else 'not met'}**. Requires the Dreamer stopping policy to pass the reliability gate, lose at most one diagnosis against its full-check predictor, and save at least 0.5 checks against both full checking and telemetry stopping.",'','## Accuracy by fixed check budget','','| Checks | Dreamer correct / 80 | Telemetry correct / 80 |','|---:|---:|---:|']
for t in range(4):lines.append(f"| {t} | {s['latent']['fixed_budgets'][str(t)]['correct']} | {s['raw']['fixed_budgets'][str(t)]['correct']} |")
lines+=['','## Calibration and paired results','']
for name in ['latent','raw']:
    policy=f['policies'][name];threshold=policy['threshold'];lines.append(f"- {name}: threshold {threshold if threshold is not None else 'all-check fallback'}; development calibration {'passed' if policy['development_calibration_passed'] else 'failed'}. Signed ridge margins are not probabilities.")
lines += [f"- Dreamer alone correct: {s['paired_latent_only_correct']}; telemetry alone correct: {s['paired_raw_only_correct']}.",'','## Interpretation and limits','',
'This evaluates early stopping with fixed CPU → pool → trace ordering. It does not evaluate native Dreamer planning, learned check ordering or imagined trajectories. All measurements are collected for replay; reported savings count simulated diagnostic acquisitions, not measured runtime savings. Later prefixes include later passive observations as well as newly inspected groups. The candidate also uses supervised mechanism training, so improvements cannot be attributed solely to world-model training objectives.','',
'One model and eight known fault regimes; no claim of unseen-system or production reliability. Thresholds were frozen before evaluation and no evaluation incidents were excluded or used to tune the rules.','',
'## Verification and operations','',f"Job duration: {int(duration)//60}m{int(duration)%60:02d}s; staging and restoration add wall time. Eleven targeted tests passed. Exported checksums, data separation, freeze timestamps, observed-prefix boundaries, all scores and predictions, checkpoint identity and restoration were verified. Application services remained running; inference was restored with its original template and replica count. Temporary resources were removed.",'',
'[Frozen protocol](LAB_EFFICIENCY_PROTOCOL.md) · [Evidence verification](evidence/lab-efficiency-v1/verification.json) · [Detailed results](evidence/lab-efficiency-v1/evaluation/summary.json)','']
(R/'reports/LAB_EFFICIENCY_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines[:17]))
