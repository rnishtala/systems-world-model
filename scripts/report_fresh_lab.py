import json
from pathlib import Path
from datetime import datetime

root=Path(__file__).resolve().parents[1];e=root/'reports/evidence/lab-fresh-v1'
v=json.loads((e/'verification.json').read_text());assert v['verified']
s=v['summary'];training=json.loads((e/'models/training-51.json').read_text())
methods={'latent':'Dreamer state + ridge (primary)','raw':'Full observation history + ridge (primary baseline)','raw_last':'Last observed telemetry + ridge','shuffled_latent':'Shuffled-label latent control','dreamer_auxiliary':'Dreamer supervised head','dreamer_reward':'Dreamer imagined reward'}
lines=['# Fresh lab diagnosis comparison','', '**Outcome: '+('added-value signal met.' if s['added_value_signal'] else 'no prespecified added-value signal.')+'**','',
'Fresh physical incidents from checkout → payment → ledger → PostgreSQL: 80 training and 80 later evaluation cases. Ten cases per healthy/single/pair/triple fault family. Training mixed ordinary and shifted load/severity; evaluation used shifted settings and a different scenario seed. All data stayed in the isolated lab.','',
'Actual upstream DreamerV3 trained from scratch for 100,000 steps (seed 51) with training-only mechanism supervision. Its state readout and the full-history telemetry baseline used the same observation sequence, penalty-1 ridge fitting, and threshold zero. Both readouts were frozen before evaluation collection. CPU, pool and trace checks were fixed, not learned.','',
'| Method | Correct / 80 | Reliability gate |','|---|---:|---|']
for name,label in methods.items():
    m=s['methods'][name];lines.append(f"| {label} | {m['correct']} | {'Pass' if m['gate_passed'] else 'Fail'} |")
lines+=['','Reliability required healthy 10/10 and at least 9/10 exact diagnoses in every other family. Added value additionally required Dreamer to exceed the full-history baseline by at least four correct incidents.','',
         f"Paired disagreements: Dreamer alone correct **{s['paired_latent_wins']}**; full-history telemetry alone correct **{s['paired_raw_wins']}**.",'',
         '| Fault family | Dreamer state / 10 | Full-history raw / 10 |','|---|---:|---:|']
for f,label in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']):
    lines.append(f"| {label} | {s['methods']['latent']['families'][str(f)]['correct']} | {s['methods']['raw']['families'][str(f)]['correct']} |")
lines+=['','## Interpretation','',
'No production or unseen-system reliability claim follows from this small known-regime sample. This is one training initialization and one fresh scenario batch. Mechanism supervision is part of the Dreamer candidate, so any advantage cannot be attributed solely to its world-model objectives without a matched supervised ablation. Conversely, a tie or loss does not prove all world-model approaches fail. The existing serving checkpoint was not promoted or modified.','',
'## Provenance and execution','',
f"Training took {training['seconds']/60:.1f} minutes. Collection, extraction, export and restoration are additional time. Thirteen targeted tests passed. All 160 cases passed clean measurement-window and request-success checks. Training and evaluation have no duplicate trajectories; evaluation timestamps follow readout freezing. Actual opposite-label policy invariance passed, and the evaluated checkpoint did not change.",'',
'Only the isolated inference deployment was paused for quota and restored to its original template/replicas. Fault controls were reset; temporary experiment resources were removed. All original held-out RCAEval results remain frozen.','',
'[Protocol](LAB_FRESH_SPLIT_PROTOCOL.md) · [Verified results](evidence/lab-fresh-v1/verification.json) · [Restoration](evidence/lab-fresh-v1/restored.json)']
(root/'reports/LAB_FRESH_SPLIT_RESULTS.md').write_text('\n'.join(lines)+'\n')
p=root/'README.md';readme=p.read_text();at=readme.index('## Current deployment')
paragraph=f"""A **fresh live-lab split** trained Dreamer on 80 new incidents and evaluated on
80 later incidents after freezing both diagnostic readouts. Dreamer state scored
**{s['methods']['latent']['correct']}/80** versus **{s['methods']['raw']['correct']}/80** for the same full observation history with a raw-telemetry
ridge classifier. Prespecified added-value signal: **{'met' if s['added_value_signal'] else 'not met'}**. This is one
model and known regimes, not production reliability. Serving remains unchanged.
[Fresh lab results](reports/LAB_FRESH_SPLIT_RESULTS.md).

"""
p.write_text(readme[:at]+paragraph+readme[at:])
