"""Generate a research note from preserved live evidence, including negative results."""
from pathlib import Path
import json,statistics
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon
root=Path(__file__).resolve().parents[1];ev=root/'reports/evidence/microservice'
blue=colors.HexColor('#123B59');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#536271')
styles=getSampleStyleSheet()
for name,size,leading,color in [('TitleX',23,27,blue),('HeadX',12,16,teal),('BodyX',10,14,colors.black),('SmallX',8,11,gray)]:
 styles.add(ParagraphStyle(name=name,fontName='Helvetica-Bold' if name in ('TitleX','HeadX') else 'Helvetica',fontSize=size,leading=leading,textColor=color,spaceAfter=8))
story=[]
def p(s,style='BodyX'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'HeadX')
def table(rows,widths):
 t=Table([[Paragraph(escape(str(x)),styles['SmallX']) for x in r] for r in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,blue)]));story.append(t)
def load(name):return json.loads((ev/name).read_text())
def pct(n):return f'{100*n:.1f}%'
trials=[('Original features / native','live-dev-v1a-summary.json'),('Physical scaling / native','live-dev-v2-summary.json'),('Inspect first / familiar','live-dev-v3-summary.json'),('Inspect first / shifted','live-dev-v3-shift-summary.json')]
confirmation=load('confirmation-summary.json') if (ev/'confirmation-summary.json').exists() else None
p('SYSTEMS WORLD MODEL / LIVE MICROSERVICE STUDY','SmallX')
p('Learning a diagnosis is not<br/>learning when to inspect','TitleX')
p('DreamerV3 transfer to an isolated HTTP and PostgreSQL lab','HeadX')
p('Technical research note | 9 November 2026 | Experimental, not peer reviewed','SmallX')
h('Abstract')
p('We train the project-linked DreamerV3 implementation from scratch on empirical trajectories collected from a real four-service lab. Native actor inference initially collapses to one diagnosis. Physical feature scaling improves fresh live accuracy from 25% to 75%, but all ledger-delay cases still fail. Keeping the same trained weights and requiring three inspections before ranking imagined diagnostic rewards yields 48/48 correct diagnoses across two development batches. Predicted inspected sensor values outperform an unconditional calibration-mean baseline in both batches. These findings support a bounded learned diagnostic with an evidence-gathering harness; they do not establish autonomous active sensing or general causal understanding.')
p('Fresh-seed confirmation: '+('passed the frozen numerical gates.' if confirmation and confirmation['frozen_confirmation_passed'] else 'did not pass the frozen numerical gates.' if confirmation else 'pending; development results are not confirmation.'))
h('Architecture and research question')
d=Drawing(490,105)
for x,w,title,sub in [(0,150,'Live microservices','HTTP + real SQL workload'),(170,150,'DreamerV3 world model','learned latent dynamics'),(340,150,'Evidence-first harness','3 checks, learned ranking')]:
 d.add(Rect(x,43,w,45,rx=5,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue));d.add(String(x+7,69,title,fontName='Helvetica-Bold',fontSize=9,fillColor=blue));d.add(String(x+7,52,sub,fontName='Helvetica',fontSize=8,fillColor=gray))
for x in (170,340):d.add(Line(x-20,65,x,65,strokeColor=teal));d.add(Polygon([x,65,x-5,68,x-5,62],fillColor=teal,strokeColor=teal))
d.add(String(0,20,'Forecast before inspection -> measure -> update latent state -> rank diagnoses.',fontName='Helvetica',fontSize=9,fillColor=gray));story.append(d)
p('Question: can learned system representations diagnose fresh real incidents after collecting evidence? The successful development controller uses a fixed inspection order; the native actor remains a separate unsuccessful condition. There is no LLM, BayesNet, repair loop or game-trained checkpoint.')
h('Isolation')
p('All resources and bounded fault injection are confined to systems-world-model-lab on AMD Halo, CPU only. Aggregate limits remain four CPUs and 6 GiB. Default-deny policies restrict service traffic, allow DNS, and prevent the inference pod from accessing fault controls. Pods have no Kubernetes credentials or host mounts. Inference mounts checkpoints and runtime read-only, without calibration data. Shared-host resource contention is still possible.')
story.append(PageBreak())
h('1. Environment, data and training')
p('The call chain is checkout -> payment -> ledger -> PostgreSQL. Requests perform actual HTTP calls and SQL queries. A query hold creates connection occupancy; faults add bounded payment CPU work, reduce the ledger connection pool, or add ledger delay. This is a synthetic banking-style workload, not a financial transaction implementation. Controls use expiring leases and runner cleanup. Unexpected request failures or expired measurement windows invalidate trials.')
table([['Element','Protocol'],['Calibration','80 live incidents, 20 per class; healthy, payment CPU, ledger pool, ledger delay. Four frames each; concurrency 4-6. No live test incidents enter training.'],['Training environment','Empirical trajectory resampling and sensor revelation, not a learned physics simulator or live training loop. Fourteen numeric observations; nine actions.'],['Dreamer reuse','Pinned upstream encoder, RSSM, decoder, reward/continuation heads and imagined actor-critic learning. New weights, 100,000 steps per seed, float32 CPU.'],['Exploration and rewards','Half of training episodes use randomly ordered inspections and a uniformly random diagnosis. No expert answer. -0.05/action, -0.15 additional inspection cost, +3/-3 correct/wrong, -2 unresolved.'],['Scaling intervention','Physical-v2 rescales continuous features relative to physical units; masks stay 0/1. Calibration data are unchanged.'],['Inference comparison','Native actor: 16 latent/actor particles vote. Inspect-first: CPU, pool, trace, then choose the maximum imagined diagnostic reward across four classes. Same weights; no expert diagnosis thresholds.']],[105,385])
h('2. Development results')
rows=[['Condition','Live n','Accuracy','Checks','Forecast MSE / baseline']]
for label,file in trials:
 m=load(file)['methods']['0'];pred='None' if m['prediction_mse'] is None else f"{m['prediction_mse']:.4f} / {m['baseline_mse']:.4f}"
 rows.append([label,m['evaluations'],pct(m['accuracy']),f"{m['checks']:.2f}",pred])
table(rows,[155,40,55,45,195])
p('Each batch contains six physical incidents per class. A topology-aware checklist achieved 100% diagnosis accuracy in these batches. The inspect-first method uses three checks; the checklist averages 2.25. This experiment therefore does not demonstrate superior diagnostic accuracy or fewer checks than expert rules. Forecast scores pool inspected sensor components; per-sensor scores are retained in the evidence.','SmallX')
p('With all three inspections forced on calibration data, the original world model ranked the correct diagnosis in 43.75% of cases; the scaled model reached 100%. These are training-data probes, not held-out performance. The live comparison is consistent with premature native diagnosis, but it does not isolate all effects of observation scale, stochastic inference and distribution shift.')
story.append(PageBreak())
h('3. Confirmation and uncertainty')
p('The frozen confirmation configuration uses new training seeds 10, 11 and 12 and the previously reserved live scenario seed 4200. There are 30 incidents per class at increased load/severity. All three models inspect the same 120 physical incidents in shuffled order: 360 model-incident evaluations are not 360 independent incidents. Models share calibration data. No tuning against confirmation is permitted.')
p('Per-seed gates: at least 90% exact diagnosis in each fault class; at least 95% healthy accuracy and at most 5% healthy false alarms; no more than three checks on average; median diagnosis below 60 seconds. Prediction gate: at least 30 forecasts and pooled MSE lower than calibration means. The harness changes the original native-action-selection criterion explicitly; numerical thresholds remain unchanged.')
if confirmation:
 rows=[['Seed','Accuracy','Checks','Median seconds','Forecast / baseline','Both gates']]
 for seed in (10,11,12):
  m=confirmation['methods'][str(seed)];rows.append([seed,pct(m['accuracy']),f"{m['checks']:.2f}",f"{m['median_seconds']:.2f}",f"{m['prediction_mse']:.4f} / {m['baseline_mse']:.4f}",str(m['triage_thresholds_met'] and m['prediction_thresholds_met'])])
 table(rows,[35,60,45,75,155,120])
else:p('Confirmation is pending. No fresh-seed reliability result is reported in this edition.')
p('The gates are sample-performance checks, not confidence guarantees. Even 30/30 correct cases in one class has a two-sided 95% Wilson lower accuracy bound of about 88.6%. Repeated models on shared incidents must not be pooled as independent trials. Larger and broader evaluations are needed for stronger reliability claims.')
p('Confirmation errors: seeds 10 and 12 diagnosed pool contention during ledger-delay incident 53; seed 11 diagnosed it correctly. Their pool-wait measurements differed because methods observe successive request batches. The checklist produced one false alarm on healthy incident 116 (119/120 overall). These results do not establish a statistically meaningful superiority of either method. Training completed in 25m49s wall time; live confirmation took 20m23s.')
story.append(PageBreak())
h('4. Failure boundary: unfamiliar combined faults')
p('In eight separate CPU-plus-ledger-delay incidents, both the development model and checklist returned payment CPU contention every time. They identified one present mechanism but missed the simultaneous delay; neither escalated. The evidence-first controller always selects a known class. Its reward ranking is not calibrated confidence. Unknown-fault recognition and safe abstention remain open work, not supported deployment capabilities.')
h('5. Interpretation and next experiments')
p('The current evidence supports a narrow experiment: a custom Dreamer world model can learn a useful mapping from inspected real-system telemetry to known fault diagnoses. It is more than an embedding because its learned transition, observation and reward heads support forecasts and imagined action outcomes. However, the empirical training surrogate, small topology, four classes, fixed checks and weak forecast baseline prevent claims of general systems understanding. The model does not predict arbitrary repairs or recover the service.')
p('Next work should test unseen topology and workloads, compare forecasts against stronger persistence and conditional baselines, train/evaluate abstention on separately held-out unfamiliar faults, and learn inspection selection without losing evidence quality. LLM explanation or action execution should remain outside the model until these boundaries are explicit and verified.')
h('Reproducibility and references')
p('Source and episode evidence: github.com/rnishtala/systems-world-model, reports/evidence/microservice. Prospective protocol: reports/MICROSERVICE_PROTOCOL.md. Failed trials are retained. The historical queue-simulator POC is a separate experiment and is not evidence of live transfer.','SmallX')
p('[1] DreamerV3 project: https://danijar.com/project/dreamerv3/. [2] Project-linked source: https://github.com/danijar/dreamerv3, revision e01491fad6434b2245a3b8ca201dd7faedcc458c. Upstream implementation is reused; this report introduces no new reinforcement-learning algorithm.','SmallX')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(gray);canvas.drawString(50,25,'Systems World Model | Live microservice experiment');canvas.drawRightString(545,25,str(doc.page))
out=root/'output/pdf/dreamerv3-live-microservice-study.pdf';out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=(595,842),rightMargin=50,leftMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
