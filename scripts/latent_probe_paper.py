"""Render the frozen-state diagnostic study from preserved results."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/latent-probe'
s=json.loads((E/'summary.json').read_text());fit=json.loads((E/'fitted.json').read_text());styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontSize=21,leading=25,spaceAfter=12,textColor=colors.HexColor('#123B59')))
styles.add(ParagraphStyle(name='BodyX',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='HeadX',fontSize=12,leading=16,spaceBefore=7,spaceAfter=7,textColor=colors.HexColor('#087E8B')))
story=[]
def p(x,style='BodyX'):story.append(Paragraph(x,styles[style]))
def table(rows,widths):
 t=Table(rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
p('Does Dreamer retain the fault signals?','TitleX')
p('Frozen latent-state probes | Systems World Model | 10 November 2026')
p('Question','HeadX')
p('The previous combined-fault experiment found 49/80 exact diagnoses with Dreamer versus 76/80 using small classifiers on identical telemetry. This follow-up asks whether secondary faults remain recoverable in Dreamer\'s internal state, despite its incomplete reward-based diagnoses. No Dreamer weights were retrained.')
p('Method','HeadX')
p(f'Three independent binary ridge probes were fitted on {fit["latent_dimensions"]} latent features: the mean of 16 particles\' deterministic and stochastic states after three fixed inspections. Identical probes used eight raw telemetry features. A shuffled-training-label latent probe provides a negative control. All probes use training-only standardization, ridge penalty 1 and threshold zero; no tuning.')
p('The original 140 calibration incidents supplied all fitting data. Probes were saved before collecting 80 fresh live incidents (seed 6400): ten each for healthy, three singles, three pairs and the unseen triple. Each incident used two warmup batches and four recorded HTTP/PostgreSQL frames. Model states and diagnoses were replayed offline after faults were reset. Hidden intervention labels never entered model observations.')
p('Fresh paired results','HeadX')
methods=['dreamer_reward','raw','latent','shuffled_latent'];names=['Dreamer reward','Raw linear','Latent linear','Shuffled latent']
rows=[['Method','Exact sets','Family gate']]
for k,n in zip(methods,names):
 m=s['methods'][k];rows.append([n,f'{m["correct"]}/{m["n"]}', 'Pass' if m['gate_passed'] else 'Fail'])
table(rows,[270,110,110])
rows=[['Family','Dreamer','Raw','Latent','Shuffled']]
for i,n in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple (held out)']):rows.append([n,*[str(s['methods'][k]['families'][str(i)]['correct'])+'/10' for k in methods]])
table(rows,[170,80,80,80,80])
story.append(PageBreak())
p('Interpretation','HeadX')
p((E/'interpretation.txt').read_text())
p('The exploratory gate requires 10/10 healthy and at least 9/10 per other family. This is one frozen model and one small test batch, not a production-reliability estimate. A failed linear probe cannot prove information is absent: nonlinear encoding, particle averaging, regularization and limited data remain alternatives. The shuffled-label control is one permutation, not a formal significance test. Offline inference does not establish online end-to-end latency.')
p('The raw probe uses the eight prespecified load/inspection features from the previous baseline. The latent state also reflects observation history, front-end latency/error signals and inspection masks, and has more dimensions. This is a decodability experiment, not a controlled proof that latent features outperform all raw-telemetry models.')
p('Isolation and reproducibility','HeadX')
p('The checkpoint digest is unchanged. All workloads stayed within systems-world-model-lab on AMD Halo under the existing 4 CPU/6 GiB cap. Only inference paused; the previous controller was restored afterward. No LLM, autonomous repair or other namespaces are involved. Protocol, latent arrays, fitted probes, predictions and logs are preserved in reports/evidence/latent-probe and reports/LATENT_PROBE_PROTOCOL.md.')
p('DreamerV3 implementation: github.com/danijar/dreamerv3, revision e01491fad6434b2245a3b8ca201dd7faedcc458c. Technical research note, not peer reviewed.')
def footer(c,d):
 c.setFont('Helvetica',8);c.drawString(50,25,'Systems World Model | Frozen latent probes');c.drawRightString(545,25,str(d.page))
out=R/'output/pdf/dreamerv3-latent-probes.pdf'
SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
