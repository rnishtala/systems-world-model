"""Research note from independently verified latent-readout confirmation evidence."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/latent-confirmation'
c=json.loads((E/'completion.json').read_text());v=json.loads((E/'verified.json').read_text());m=json.loads((E/'eval-23/summary.json').read_text())['methods'];t=json.loads((E/'training-23.json').read_text());primary=m['latent']
st=getSampleStyleSheet()
for name,size,leading,color in [('T',20,24,'#123B59'),('H',12,16,'#087E8B'),('B',9.5,12.8,'#182C3A')]:st.add(ParagraphStyle(name=name,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=7))
story=[]
def p(text,style='B'):story.append(Paragraph(text,st[style]))
def table(rows,widths):
 t=Table(rows,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
p('Independent confirmation of Dreamer-state triage','T')
p('Systems World Model | Isolated AMD Halo experiment | 10 November 2026')
p('Abstract','H')
p(f"A development experiment suggested that a classifier over DreamerV3's learned state can diagnose faults more reliably than its imagined-reward readout. We froze the classifier recipe, independently trained model seed 23, and evaluated 80 fresh physical incidents. The primary classifier scored {primary['correct']}/80; raw telemetry scored {m['raw']['correct']}/80, and imagined reward scored {m['dreamer_reward']['correct']}/80. The prespecified gate {'passed' if primary['gate_passed'] else 'failed'}. The result concerns a small service chain and eight known fault regimes, not production reliability or autonomous resolution.")
p('Hypothesis and development evidence','H')
p('After removing warm-up contamination, development model 22 achieved 80/80 with a latent-state ridge classifier, versus 77/80 through imagined rewards. Candidate selection followed that observation; therefore those 80 incidents are development evidence. This study uses a new model initialization and a new scenario batch. Its primary outcome was explicitly changed to the latent classifier before training and collection. Earlier failed primary gates remain failed.')
p('System and learned-state architecture','H')
table([['Training on calibration only','Frozen confirmation inference'],['Checkout -> payment -> ledger -> PostgreSQL','Two warm-up batches -> clear metric rows'],['160 recorded incidents, all eight regimes','Four measured frames, fixed inspections'],['Dreamer encoder + recurrent world model','Mean of 16 posterior-state particles'],['Original losses + auxiliary fault-label loss','Frozen three-label ridge classifier'],['Fit classifier in the new latent coordinates','Threshold at zero -> exact fault set']], [247,247])
p('The actual upstream DreamerV3 implementation is used, with a 1M-size configuration, CPU float32 and unchanged auxiliary-loss weight 1. Model 23 is trained from scratch for 100,000 steps. The original 160 calibration incidents are reused unchanged; no held-out incident enters training. Fault labels are training targets only and are excluded from the encoder, decoder and inference policy input.')
p('Frozen classifier and acceptance rule','H')
p('The candidate concatenates deterministic and stochastic recurrent state, averages 16 particles, standardizes using calibration features, and fits signed fault labels with ridge penalty 1.0. Threshold 0.0 independently selects CPU, pool and delay bits. The recipe, not model-22 coefficients, is frozen: a new model needs fitting in its own latent coordinates. No test-time refitting is allowed.')
p('Fresh scenario seed 6801 supplies ten incidents per family at shifted load/severity. Acceptance requires 10/10 healthy and at least 9/10 exact fault sets in every other family. All requests must succeed, incident duration must be at most 35 seconds, and all three services must report exactly n, 2n, 3n and 4n measured requests. Raw telemetry, reward, auxiliary and shuffled-label latent readouts are secondary comparisons.')
story.append(PageBreak())
p('Confirmation results','T')
p(f"Training completed in {t['seconds']/60:.2f} minutes. All 80 fresh measurement windows passed validity checks. Checkpoint weights remained unchanged through evaluation.")
table([['Readout','Exact diagnoses','Gate']]+[[name,str(m[name]['correct'])+'/80','Pass' if m[name]['gate_passed'] else 'Fail'] for name in ('latent','raw','dreamer_reward','dreamer_auxiliary','shuffled_latent')],[240,130,124])
names=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']
table([['Fault regime','Latent primary','Telemetry','Reward']]+[[name]+[str(m[k]['families'][str(i)]['correct'])+'/10' for k in ('latent','raw','dreamer_reward')] for i,name in enumerate(names)],[206,112,88,88])
p('Interpretation and limits','H')
p(('The independent acceptance gate passed, supporting a limited known-regime triage POC.' if primary['gate_passed'] else 'The independent acceptance gate failed; the development result did not reproduce an acceptance pass.')+' This candidate uses Dreamer to learn a recurrent representation, then a supervised classifier for the final diagnosis. It does not demonstrate that imagined rollout rewards reliably select diagnoses, or that the model has learned how to investigate an incident: inspections are fixed.')
p('One model repeat and ten incidents per regime cannot establish production reliability or superiority over telemetry classification. All combinations are represented in calibration. Historical training telemetry includes warm-up samples, whereas evaluation deliberately excludes them; startup incidents and unseen services need separate tests. LLM orchestration and remediation actions have not been evaluated.')
p('Decision and next step','H')
p(('Next: implement the frozen readout in the inference-only API, verify parity against preserved predictions, and evaluate fresh live incidents through that API before promotion.' if primary['gate_passed'] else 'Next: analyze the preserved errors and calibration coverage. Any revised candidate requires a new prespecified confirmation; this batch must not become a tuning set while retaining its confirmation label.')+' Original service bundles and all five baseline replicas were restored. No candidate was promoted and no other namespace was modified.')
p('Reproducibility','H')
p('Three classifier and four monitor tests passed. Verification reproduced all probe coefficients from calibration features and all probe predictions from held-out features, checked runtime label-isolation audits, and verified checkpoint and protocol hashes. Repository: rnishtala/systems-world-model. Protocol/results: reports/LATENT_CONFIRMATION_*.md. Evidence: reports/evidence/latent-confirmation/. DreamerV3 upstream revision: e01491fad6434b2245a3b8ca201dd7faedcc458c (github.com/danijar/dreamerv3).')
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607381'));canvas.drawString(50,27,'Systems World Model | Experimental results, not production validation');canvas.drawRightString(545,27,str(doc.page))
SimpleDocTemplate(str(R/'output/pdf/dreamerv3-latent-confirmation.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
