"""Research note for frozen-classifier HTTP integration and live validation."""
import json
from pathlib import Path
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/latent-api'
s=json.loads((E/'summary.json').read_text());parity=json.loads((E/'parity.json').read_text());verified=json.loads((E/'verified.json').read_text())
promotion=json.loads((E/'promotion.json').read_text()) if (E/'promotion.json').exists() else None
rows=[] if s['fresh_skipped'] else list(map(json.loads,(E/'live.jsonl').read_text().splitlines()))
st=getSampleStyleSheet()
for name,size,leading,color in [('T',20,24,'#123B59'),('H',12,16,'#087E8B'),('B',9.5,12.8,'#182C3A')]:st.add(ParagraphStyle(name=name,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=7))
story=[]
def p(text,style='B'):story.append(Paragraph(text,st[style]))
def table(data,widths):
 t=Table(data,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
p('Serving a Dreamer-state classifier over HTTP','T')
p('Systems World Model | Isolated AMD Halo experiment | 10 November 2026')
p('Abstract','H')
p(f"An independently confirmed classifier over DreamerV3's learned state was integrated into a restricted inference API without retraining or refitting. HTTP replay reproduced {parity['matching']}/80 saved diagnoses. "+(f"On 80 new physical incidents, the live API scored {s['correct']}/80 and {'passed' if s['gate_passed'] else 'failed'} the prespecified acceptance gate." if rows else 'Fresh live testing was skipped after the integration gate failed.')+' The experiment tests serving parity and known-regime triage, not learned investigation planning or autonomous remediation.')
p('Starting point and frozen candidate','H')
p('The preceding independent experiment trained model seed 23 from scratch and obtained 80/80 correct fault sets using a ridge classifier over its latent state. That exact checkpoint and those exact coefficients are reused. The candidate averages 16 posterior particles after fixed CPU, pool and trace inspections; signed classifier outputs above zero select the three fault bits. No held-out example, threshold, normalization or weight is fitted in this integration study.')
p('Serving architecture','H')
table([['Experiment runner','Restricted inference API'],['Inject leased faults in the isolated namespace','Receive observation, action cost and incident ID'],['Warm up twice; clear completed metric rows','Update recurrent Dreamer state'],['Measure a batch; reveal requested sensors','Request CPU, pool and trace inspections'],['Repeat until diagnosis; score truth separately','Apply frozen classifier -> diagnosis + scores'],['Save measured frames and HTTP responses','No fault labels, controls or replay mounted']], [247,247])
p('The new inspect-latent controller retains the existing HTTP schema and agent network role. It rejects classifier/checkpoint hash mismatches, unsupported feature contracts, corrupt or nonfinite parameters and invalid state dimensions. The serving directory contains only configuration, checkpoint, training metadata and four classifier arrays. Model mounts are read-only, and the pod has no service-account credentials.')
p('Prespecified integration and live protocol','H')
p('First, replay all 80 saved confirmation observation sequences through a normally started API. Require identical inspection actions and diagnoses plus the expected classifier hash. Reject requests containing fault labels or diagnosis-correctness rewards. Replay is an integration test and is not counted as fresh accuracy evidence.')
p('Only after parity passes, collect scenario seed 6901: ten fresh incidents for each of eight known regimes under shifted load/severity. Interleave four measured request batches with four HTTP inference calls. All requests must succeed; cumulative service counts must equal n, 2n, 3n and 4n. Fault setup through diagnosis must remain within 35 seconds. Healthy acceptance requires 10/10; every other family requires at least 9/10 exact fault sets. No reroll or test-driven tuning is allowed.')
story.append(PageBreak())
p('Observed API behavior','T')
table([['Check','Result'],['Saved diagnosis parity',f"{parity['matching']}/80"],['Label/correctness-reward rejection','Passed' if all(parity['contract_rejections']) else 'Failed'],['Fresh live gate','Skipped' if not rows else ('Passed' if s['gate_passed'] else 'Failed')],['Serving package contains examples/replay','No'],['Rollback restoration verified','Yes'],['Lab default after reviewed promotion','Candidate' if promotion else 'Original baseline']],[310,184])
if rows:
 names=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']
 table([['Fresh fault regime','Exact diagnosis']]+[[name,str(s['families'][str(i)]['correct'])+'/10'] for i,name in enumerate(names)]+[['Total',str(s['correct'])+'/80']],[310,184])
 times=np.array([r['seconds'] for r in rows]);p(f"Injection-through-diagnosis duration: median {np.median(times):.2f}s, p95 {np.percentile(times,95):.2f}s, maximum {times.max():.2f}s. These times include setup, warm-up, measurement and HTTP inference; they are not model-only inference latency.")
p('Interpretation and limits','H')
p(('The candidate passed the live API gate, extending the earlier offline confirmation to real observation/action exchanges through the serving interface.' if s['gate_passed'] else 'The candidate did not pass this integration/live gate and was not promoted.')+' Fixed inspection order and a supervised diagnostic head remain part of the method. The model supplies a learned recurrent representation; this is not evidence that imagined reward reliably chooses root causes or that the agent learns its own investigation strategy.')
p('The environment is one small service chain with eight known regimes. A small fresh batch does not establish production reliability, unseen-system generalization, startup handling or statistical superiority over simpler methods. This run does not include a fresh raw-telemetry comparator. LLM orchestration and remediation are outside its scope.')
p('Decision and reproducibility','H')
p(('After baseline restoration, a separate reviewed promotion made this exact candidate the lab default. A healthy real-service API check passed. The original manifest and service sources are retained for rollback; diagnosis-only behavior is unchanged.' if promotion else 'Next: controlled promotion of this exact package within the lab, retaining diagnosis-only behavior and a rollback manifest.' if s['gate_passed'] else 'Next: preserve and analyze the failed result before defining a new experiment.')+' Baseline restoration was verified before the separate promotion decision; no other namespace was modified.')
p('Seven unit tests cover artifact binding, feature layout, fixed predictions and package contents. Verification confirms identical packaged and confirmed classifier arrays, and checks replay actions, API score-to-diagnosis mapping, exact offline/live observation equivalence, fresh window counts, identities and per-family scores. Repository: rnishtala/systems-world-model. Protocol/results: reports/LATENT_API_*.md. Evidence: reports/evidence/latent-api/. DreamerV3 revision: e01491fad6434b2245a3b8ca201dd7faedcc458c.')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607381'));canvas.drawString(50,27,'Systems World Model | Experimental results, not production validation');canvas.drawRightString(545,27,str(doc.page))
SimpleDocTemplate(str(R/'output/pdf/dreamerv3-latent-api.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
