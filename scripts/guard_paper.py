"""Report the completed guard pilot without rewriting the earlier confirmation."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing,Rect,String,Line
root=Path(__file__).resolve().parents[1];result=json.loads((root/'reports/evidence/guard/summary.json').read_text());blue=colors.HexColor('#123B59');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#536271')
styles=getSampleStyleSheet()
for name,size,leading,color in [('TitleX',22,26,blue),('HeadX',12,16,teal),('BodyX',10,14,colors.black),('SmallX',8,11,gray)]:styles.add(ParagraphStyle(name=name,fontName='Helvetica-Bold' if name in ('TitleX','HeadX') else 'Helvetica',fontSize=size,leading=leading,textColor=color,spaceAfter=8))
story=[]
def p(s,style='BodyX'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'HeadX')
def pct(x):return f'{100*x:.1f}%'
def table(rows,widths):
 t=Table([[Paragraph(escape(str(v)),styles['SmallX']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
p('SYSTEMS WORLD MODEL / ESCALATION FOLLOW-UP','SmallX')
p('Can prediction errors reveal<br/>an unfamiliar incident?','TitleX')
p('A frozen forecast-error guard around DreamerV3 | 10 November 2026','SmallX')
h('Result')
p('The predictive guard '+('passed' if result['passed'] else 'did not pass')+' the prespecified pilot gates. Three previously trained DreamerV3 models were tested on 60 fresh known-case incidents and 36 combined-fault incidents, shared across models. No model weights were updated. This experiment tests a harness intervention, not a new world model or an independently trained replication.')
h('Known-case diagnosis and false escalation')
rows=[['Seed','Healthy','CPU','Pool','Delay','False escalation']]
for seed,m in result['seeds'].items():rows.append([seed,*[pct(m['known_classes'][str(i)]['accuracy']) for i in range(4)],pct(m['known_false_escalation_rate'])])
table(rows,[40,75,75,75,75,150])
p('Each known class has 15 incidents per seed. Escalations count as unsuccessful known-case diagnoses. These small samples do not prove production reliability. The same telemetry supports both the underlying candidate and the guard decision; this avoids rerunning a duplicate live experiment.','SmallX')
p('Underlying candidate accuracy before applying the guard: '+', '.join(seed+': '+pct(m['underlying_candidate_accuracy']) for seed,m in result['seeds'].items())+'. This distinguishes errors in learned diagnosis from rejections introduced by the guard. Fresh known-case failures also limit the broader reliability claim; passing the earlier frozen batch did not establish robustness across every subsequent incident stream.','SmallX')
h('Combined-fault escalation')
rows=[['Seed','CPU + delay','CPU + pool','Pool + delay','All gates']]
for seed,m in result['seeds'].items():rows.append([seed,*[pct(m['combined'][n]['escalation_rate']) for n in ('cpu-delay','cpu-pool','pool-delay')],str(m['passed'])])
table(rows,[40,115,115,115,105])
p('Each combination has 12 physical incidents, inspected by all three models. A single known-class answer is incomplete for these injections. Escalation is a useful rejection of the single-fault explanation; it does not identify both causes or demonstrate recovery.','SmallX')
h('Deployment decision')
p('The guarded controller is retained for this isolated experiment.' if result['passed'] else 'The confirmed inspect-first controller is restored. The unsuccessful guard, fitted parameters, protocol and raw observations are retained for research; it is not promoted as an improved default. Restoration preserves prior behavior; it does not fix the fresh known-class errors.')
story.append(PageBreak())
h('Method: a harness change around frozen weights')
p('Earlier experiments showed that combined faults can receive imagined diagnostic rewards as high as ordinary known faults. Those rewards cannot be treated as calibrated confidence. We instead compare telemetry with the world model\'s forecasts, emitted before each inspection.')
d=Drawing(490,90)
for x,title,sub in [(0,'Forecast','before inspection'),(165,'Measure','actual sensor values'),(330,'Compare','reject contradictions')]:
 d.add(Rect(x,30,155,42,rx=4,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue));d.add(String(x+8,55,title,fontName='Helvetica-Bold',fontSize=10,fillColor=blue));d.add(String(x+8,39,sub,fontName='Helvetica',fontSize=8,fillColor=gray))
for x in (155,320):d.add(Line(x,51,x+10,51,strokeColor=teal))
story.append(d)
p('For each seed, true known class and sensor, the fitted limit is 1.5 times the largest absolute forecast error in the earlier confirmation data, plus 0.05 normalized units. At inference, the highest imagined-reward class chooses the applicable limits. If any of seven inspected sensor errors exceeds its limit, the harness escalates. It receives fitted parameters, not hidden incident labels or raw calibration episodes.')
h('Data separation and acceptance')
p('The earlier 120-incident confirmation is explicitly repurposed as calibration for this NEW study. Its previous success remains valid for the old controller, but cannot validate the new guard. Coefficients and pilot seeds were fixed before fresh outcomes were observed. Fresh known seed: 5200; combined seeds: 5300, 5400, 5500. All use increased load/severity. No threshold tuning against this pilot is permitted.')
p('Each model must retain at least 90% exact accuracy in each known fault class and 95% healthy accuracy, with no more than 5% false escalation across known incidents. Each combined family requires at least 90% escalation. Median diagnosis must remain below 60 seconds with at most three inspections. Gates are per seed; averages cannot hide a failing model.')
h('Interpretation and limits')
p('Maximum-error envelopes are empirical bounds, not conformal intervals or confidence guarantees. Conservative limits can preserve known-case coverage while missing unfamiliar cases. Conversely, narrower limits may reject valid workload shifts. Successful rejection on three injected combinations would still not establish detection of arbitrary unknown failures. A closed set of diagnostic rewards may also remain confident when several mechanisms coexist.')
p('The follow-up should separate independent fault mechanisms in training and test abstention on fault families held out from fitting. That would change the learned model and its training data. It must be evaluated against fresh incidents and the existing checklist, not justified only by improvements on these already-observed cases.')
h('Isolation and reproducibility')
p('All code and artifacts remain in the standalone systems-world-model repository. All service traffic and leased fault injection remain inside systems-world-model-lab on AMD Halo; aggregate cap: 4 CPU / 6 GiB. Control port access remains unavailable to inference. No Kubernetes token, host mount, LLM integration or automated repair is introduced.','SmallX')
p('Protocol: reports/UNKNOWN_FAULT_PROTOCOL.md. Exact per-seed results, episode traces, source hash and job log: reports/evidence/guard/. Frozen base experiment: output/pdf/dreamerv3-live-microservice-study.pdf. Upstream implementation: github.com/danijar/dreamerv3, revision e01491fad6434b2245a3b8ca201dd7faedcc458c. Technical note, not peer reviewed.','SmallX')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(gray);canvas.drawString(50,25,'Systems World Model | Predictive escalation pilot');canvas.drawRightString(545,25,str(doc.page))
out=root/'output/pdf/dreamerv3-predictive-escalation.pdf'
SimpleDocTemplate(str(out),pagesize=(595,842),rightMargin=50,leftMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
