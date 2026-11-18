"""Render verified development and fresh confirmation evidence without retuning."""
import json
from pathlib import Path
from datetime import datetime
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/lab-order-confirm-v1'
s=json.loads((O/'evaluation/summary.json').read_text());assert json.loads((O/'verification.json').read_text())['verified']
job=json.loads((O/'job.json').read_text());secs=(datetime.fromisoformat(job['completionTime'].replace('Z','+00:00'))-datetime.fromisoformat(job['startTime'].replace('Z','+00:00'))).total_seconds()
order=lambda key:' → '.join({'1':'CPU','2':'pool','3':'trace'}[x] for x in key)
l=s['latent'];r=s['raw'];ls=l['stopping'];rs=r['stopping']
checks=[('Reliability: healthy 10/10, other families ≥9/10',ls['gate_passed']),('At most one fewer correct diagnosis than chosen-order full checking',ls['correct']>=l['full']['correct']-1),('Save ≥0.5 mean checks versus three checks',3-ls['mean_checks']>=.5),('Save ≥0.5 mean checks versus telemetry stopping',rs['mean_checks']-ls['mean_checks']>=.5)]
lines=['# Fresh confirmation: development-selected check orders','',f"**Prespecified confirmation gate: {'passed' if s['confirmation_passed'] else 'failed'}.**",'',
'Policies were fitted and frozen using scenario 8201 development data, then tested on 80 fresh scenario 8401 incidents (ten per known fault family). Neither policy changed during evaluation. This is a fresh confirmation of the fitted policies, separate from the preceding nested development result.','',
'| Method | Frozen check order | Correct / 80 | Mean checks / 3 | Reliability gate |','|---|---|---:|---:|---|']
for name,label in [('latent','Dreamer state'),('raw','Telemetry history')]:
 m=s[name];p=m['stopping'];lines.append(f"| {label}, stopping | {order(m['order'])} | {p['correct']} | {p['mean_checks']:.3f} | {'Pass' if p['gate_passed'] else 'Fail'} |")
 p=m['full'];lines.append(f"| {label}, all checks | {order(m['order'])} | {p['correct']} | 3.000 | {'Pass' if p['gate_passed'] else 'Fail'} |")
lines+=['','## Prespecified criteria','']+[f"- {'Pass' if passed else 'Fail'}: {title}." for title,passed in checks]
lines+=['',f"Paired correctness: Dreamer alone correct {s['paired_latent_only_correct']}; telemetry alone correct {s['paired_raw_only_correct']}.",'','## Per-family stopping accuracy','','| Family | Dreamer / 10 | Telemetry / 10 |','|---|---:|---:|']
for f,label in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']):lines.append(f"| {label} | {ls['families'][str(f)]['correct']} | {rs['families'][str(f)]['correct']} |")
lines+=['','## Interpretation and limits','',
'The earlier development selection scored Dreamer 80/80 at 2.425 checks versus telemetry 78/80 at 2.975. Those numbers are not confirmation results. The final fit selected one order per method using all development cases; it did not choose the best order from this fresh batch. Readouts are supervised classifiers over a frozen Dreamer state or matched raw history. This does not demonstrate native Dreamer planning or isolate the causal benefit of world-model objectives.','',
'Checks have unit cost. The experiment records complete trajectories and replays observation acquisition, so check savings are simulated diagnostic acquisitions, not measured wall-time savings. Reordering assumes passive inspections do not change system dynamics. This is one model and a small known-regime batch, not unseen-system or production reliability. No model was promoted.','',
'## Verification and operations','',f"Job duration: {int(secs)//60}m{int(secs)%60:02d}s; staging and restoration add wall time. An initial SSH transfer timeout occurred before job creation or any collection; its log is preserved. No collected incident was rerun or excluded. Seventeen targeted tests passed.",'',
'Verification reproduced final policy selection and every fitted coefficient from development data, then every fresh evaluation score and stopping decision. It checked source/checkpoint hashes, freeze-before-collection timestamps, clean measurement windows, distinct development/evaluation trajectories, observed-prefix boundaries and unchanged model/parameters. Inference was restored with its original template and replica count; temporary resources were removed. All operations stayed in systems-world-model-lab.','',
'[Development results](LAB_CHECK_ORDER_RESULTS.md) · [Frozen confirmation protocol](LAB_ORDER_CONFIRM_PROTOCOL.md) · [Verification](evidence/lab-order-confirm-v1/verification.json) · [Detailed results](evidence/lab-order-confirm-v1/evaluation/summary.json)','']
(R/'reports/LAB_ORDER_CONFIRM_RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines[:14]))
