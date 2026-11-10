"""Generate the research note from the preserved development evidence."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/mechanisms'
s=json.loads((E/'summary.json').read_text());m=s['models']['20'];baseline=json.loads((E/'baseline-summary.json').read_text())['models']['binary-trees'];training=json.loads((E/'training-20.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontSize=22,leading=26,textColor=colors.HexColor('#123B59'),spaceAfter=12))
styles.add(ParagraphStyle(name='BodyX',fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='HeadX',fontSize=12,leading=16,textColor=colors.HexColor('#087E8B'),spaceBefore=8,spaceAfter=7))
story=[]
def p(t,style='BodyX'):story.append(Paragraph(t,styles[style]))
def table(rows,widths):
    t=Table(rows,colWidths=widths)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
p('Learning simultaneous system faults','TitleX')
p('DreamerV3 compositional diagnosis development study | 10 November 2026')
p('Result: 49/80 exact diagnoses with Dreamer (61.25%), versus 76/80 (95%) with a lightweight learned comparator on identical observations. Neither passed every family gate. Dreamer missed all CPU-containing combinations; changing the answer format and adding pair training did not solve compositional diagnosis.')
p('Question and design','HeadX')
p('Can a world model distinguish connection-pool pressure from delay while also identifying simultaneous CPU contention? Earlier single-fault models sometimes confused delay and pool pressure, and could not express a combined answer. This experiment keeps upstream DreamerV3 and the telemetry adapter, but changes the diagnosis action space and retrains its weights from scratch.')
p('HTTP / SQL observations -> recurrent world model -> imagined rewards for eight fault sets -> diagnosis. Three inspections are fixed by the harness; the diagnosis ranking is learned. No LLM, Bayesian layer, or autonomous remediation is used.')
p('Method','HeadX')
p('Training uses 140 physical calibration incidents: 20 each for healthy, three individual mechanisms and three pairs. Ordinary and increased load/severity alternate. The model receives four-frame trajectories with requested telemetry only; intervention labels appear only in training rewards. Reward sums three presence/absence decisions (+1 correct, -1 incorrect). Random inspection/diagnosis coverage is used on half the training episodes. The CPU/pool/delay triple is absent from training.')
p(f'A fresh model (seed 20) trained for {training["steps"]:,} steps in {training["seconds"]/60:.1f} minutes. Evaluation uses 80 fresh physical incidents (scenario seed 6300), ten per family, at increased load/severity. Unlike the previous studies, increased severities and pairs are represented in training; these are not matched head-to-head comparisons.')
p('Live results','HeadX')
names=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','CPU + pool + delay (held out)']
rows=[['Incident family','Dreamer','Binary trees']]
for i,name in enumerate(names):
    f=m['families'][str(i)];b=baseline['families'][str(i)];rows.append([name,f'{f["correct"]}/{f["n"]}',f'{b["correct"]}/{b["n"]}'])
table(rows,[300,95,95])
p(f'Median diagnosis time: {m["median_seconds"]:.2f} seconds. Escalations: {m["escalations"]}. Development gate: '+('PASSED.' if s['passed'] else 'FAILED.')+' Required: 10/10 healthy and at least 9/10 exact sets in every other family. Invalid requests or expired fault leases invalidate the run.')
p('The comparator fits three independent binary trees (depth 3, minimum leaf 5) on calibration data only, then classifies the same live inspection traces offline. Its settings and fitted trees were frozen before live evaluation. Its classification-only timing is not an end-to-end latency comparison.')
p('Interpretation and limits','HeadX')
p('This is a small, single-seed development experiment, not a reliability confirmation. Eight discrete answer masks with additive rewards do not guarantee compositional generalization. Held-out triple behavior tests a new combination of familiar mechanisms, not an arbitrary unseen cause. The training environment replays measured trajectories; the study does not show learned intervention dynamics, repair planning, or autonomous recovery.')
p("The comparator recovered most combinations, so missing telemetry alone cannot explain Dreamer's failures. The experiment does not distinguish loss of information in the recurrent state from a weak reward readout. A post-hoc additive projection of the eight predicted rewards still gives 49/80; changing the final argmax alone did not rescue this run. This post-hoc check is diagnostic, not validation.")
p('Next experiment: fit independent CPU/pool/delay probes on frozen Dreamer latent states using training data only, then evaluate on fresh incidents. Compare against raw-telemetry probes to locate the information loss before another full retraining. The simple baseline also missed two of ten triples, so it is not a reliable replacement yet.')
p('The previous confirmed controller is restored after the experiment. All workloads remain in systems-world-model-lab on AMD Halo, capped at 4 CPU / 6 GiB; inference cannot reach fault controls. Other namespaces and repositories are outside the experiment.')
p('Reproducibility','HeadX')
p('Protocol: reports/MECHANISMS_PROTOCOL.md. Raw calibration, live traces, checkpoint metadata, job logs and scoring: reports/evidence/mechanisms/. Upstream DreamerV3 revision: e01491fad6434b2245a3b8ca201dd7faedcc458c. Source: github.com/danijar/dreamerv3. Technical research note; not peer reviewed.')
def footer(c,d):
    c.setFont('Helvetica',8);c.drawString(50,25,'Systems World Model | Compositional diagnosis');c.drawRightString(545,25,str(d.page))
out=R/'output/pdf/dreamerv3-compositional-diagnosis.pdf'
SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
