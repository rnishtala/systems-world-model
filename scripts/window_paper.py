"""Render the measured window experiment as a short reproducible research note."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/measurement-window'
c=json.loads((E/'completion.json').read_text());seeds=[22,23] if c['confirmation_passed'] is not None else [22]
results={s:json.loads((E/f'eval-{s}/summary.json').read_text())['methods'] for s in seeds}
st=getSampleStyleSheet()
for name,size,leading,color in [('T',20,24,'#123B59'),('H',12,16,'#087E8B'),('B',9.5,12.8,'#182C3A')]:st.add(ParagraphStyle(name=name,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=7))
story=[]
def p(text,style='B'):story.append(Paragraph(text,st[style]))
def table(rows,widths):
 t=Table(rows,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
m=results[22]
p('Separating warm-up from measured system triage','T')
p('Systems World Model | AMD Halo isolated experiment | 10 November 2026')
p('Abstract','H')
p(f"A prior DreamerV3 triage run failed its acceptance gate partly because warm-up latency remained in cumulative telemetry. We added a control-only measurement reset and evaluated the unchanged model on 80 fresh incidents. Imagined-reward diagnosis scored {m['dreamer_reward']['correct']}/80; the telemetry-only comparator scored {m['raw']['correct']}/80. The primary development gate {'passed' if c['development_passed'] else 'failed'}. This is an instrumentation experiment, not evidence of production reliability or a demonstrated advantage over telemetry classification.")
p('Motivation and hypothesis','H')
p('The previous complete-coverage run scored 77/80 with one healthy false positive shared by Dreamer and telemetry classification. Measured batches took about 0.06 seconds, but reported ledger p95 latency remained about 0.44 seconds because earlier warm-up requests were still included. The exact startup cause was not established. We hypothesized that explicitly separating warm-up and measurement would eliminate this contamination.')
p('Architecture and intervention','H')
table([['Physical environment','Triage path'],['Checkout -> payment -> ledger -> PostgreSQL','Four telemetry frames -> requested observations'],['CPU, pool and delay faults with bounded leases','Encoder -> recurrent Dreamer world model'],['Two complete warm-up batches','16 posterior particles -> imagined rewards'],['Clear metric rows only, then four batches','Eight diagnosis candidates -> highest reward']], [247,247])
p('The reset operates only on isolated control port 9090. It clears completed telemetry rows under the service lock; it does not rebuild connection pools, alter faults or renew leases. All warm-up requests must finish first. Each measured frame is checked for exact cumulative request counts n, 2n, 3n and 4n across all three services. Contamination invalidates the run rather than silently changing its score.')
p('Frozen experimental protocol','H')
p('Model seed 22, its 160-incident calibration and 100,000-step training checkpoint are reused unchanged. The actual pinned upstream DreamerV3 implementation is retained, with the existing auxiliary mechanism-training loss; inference inputs contain no fault labels. Primary diagnosis remains imagined-reward argmax. Auxiliary, raw-telemetry ridge, latent ridge and shuffled-latent readouts are comparators. Probes are fitted before fresh collection.')
p('Scenario seed 6701 supplies ten fresh incidents per known regime at increased load/severity. Healthy acceptance requires 10/10; every other family requires at least 9/10 exact fault sets. Only a development pass triggers independent model seed 23 and fresh scenario seed 6702. No thresholds or model weights are changed using test outcomes.')
story.append(PageBreak())
p('Results and interpretation','T')
for seed in seeds:
 m=results[seed]
 p(f"{'Development' if seed==22 else 'Independent confirmation'}: model {seed}",'H')
 table([['Readout','Exact diagnoses','Gate']]+[[name,str(m[name]['correct'])+'/80','Pass' if m[name]['gate_passed'] else 'Fail'] for name in ('dreamer_reward','dreamer_auxiliary','raw','latent','shuffled_latent')],[240,130,124])
if len(seeds)==1:
 names=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple'];m=results[22]
 table([['Fault regime','Reward','Auxiliary','Telemetry']]+[[name]+[str(m[k]['families'][str(i)]['correct'])+'/10' for k in ('dreamer_reward','dreamer_auxiliary','raw')] for i,name in enumerate(names)],[230,88,88,88])
p('Decision','H')
p(('The primary development gate failed, so independent confirmation was skipped as prespecified.' if c['confirmation_skipped'] else 'Independent confirmation '+('passed.' if c['confirmation_passed'] else 'failed.'))+' No candidate was promoted. Original service source bundles and all five baseline deployment replicas were restored; all cluster changes remained within systems-world-model-lab.')
p('Readout bottleneck and next experiment','H')
p('The primary errors all omitted pool contention: two CPU + pool cases and one triple. The frozen latent-state classifier scored 80/80 versus 8/80 with shuffled labels. This suggests useful information in the learned state, but it is a secondary result. The next experiment should prespecify that fixed readout and test an independent model on fresh incidents; it cannot retroactively rescue this failed primary gate.')
p('Calibration still includes warm-up samples: retaining it intentionally isolates a measurement-only intervention. This study covers eight known fault regimes in one small service chain. Startup transients require a separate explicit test; success on steady-state windows cannot establish startup reliability. An independent repeat, matched world-model ablation and broader workloads remain necessary. LLM-guided investigation and end-to-end remediation are future work.')
p('Reproducibility','H')
p('Four focused measurement/control tests and four monitor regression tests passed. Preserved evidence includes all fresh trajectories, predictions, checkpoint hashes, label-isolation audits and verified window counts. Source and protocol hashes precede collection. Repository: rnishtala/systems-world-model; protocol and results: reports/MEASUREMENT_WINDOW_*.md; evidence: reports/evidence/measurement-window/. DreamerV3 upstream pin: e01491fad6434b2245a3b8ca201dd7faedcc458c (github.com/danijar/dreamerv3).')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607381'));canvas.drawString(50,27,'Systems World Model | Experimental results, not production validation');canvas.drawRightString(545,27,str(doc.page))
SimpleDocTemplate(str(R/'output/pdf/dreamerv3-measurement-window.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
