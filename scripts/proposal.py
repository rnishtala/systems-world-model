"""Render the one-page research proposal; targets are prospective, not results."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

root=Path(__file__).resolve().parents[1]
blue=colors.HexColor('#123B59');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#536271')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=blue,spaceAfter=9))
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=10,leading=13.5,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=8,leading=11,textColor=gray,spaceAfter=7))
styles.add(ParagraphStyle(name='HeadX',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=teal,spaceBefore=6,spaceAfter=4))
story=[]
def p(text,style='BodyX'):story.append(Paragraph(text,styles[style]))
p('RESEARCH PROPOSAL | SYSTEMS WORLD MODEL | 9 NOVEMBER 2026','SmallX')
p('Learning to investigate system failures','TitleX')
p('<b>Goal.</b> Demonstrate that a DreamerV3 agent can choose informative diagnostic checks, distinguish failures with similar symptoms, and reach an evidence-backed diagnosis. Establish whether this helps before extending it to LLM-assisted repair.')
p('What we will build','HeadX')
p('Reuse the DreamerV3 implementation and train a new world model on system observations, actions and outcomes. The model learns a compact state and predicts future observations and rewards; its actor learns inspection choices through imagined trajectories. This is a separately trained dynamics model and policy, not LLM fine-tuning or reuse of game-trained weights.')
d=Drawing(500,78)
for x,w,title,sub in [(0,117,'Lab simulator','symptoms + outcomes'),(127,117,'DreamerV3','learn + imagine'),(254,117,'Diagnostic policy','choose next check'),(381,117,'Diagnosis','verify against truth')]:
    d.add(Rect(x,24,w,42,rx=4,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue))
    d.add(String(x+7,50,title,fontName='Helvetica-Bold',fontSize=9,fillColor=blue))
    d.add(String(x+7,34,sub,fontName='Helvetica',fontSize=7.5,fillColor=gray))
for x in (127,254,381):
    d.add(Line(x-10,45,x,45,strokeColor=teal));d.add(Polygon([x,45,x-4,48,x-4,42],fillColor=teal,strokeColor=teal))
d.add(String(0,7,'Closed loop: each selected inspection returns a new observation to the model.',fontName='Helvetica',fontSize=8,fillColor=gray))
story.append(d)
p('Smallest credible proof','HeadX')
p('Start with a bounded queue simulator: healthy operation, CPU bottleneck and downstream slowdown. The two faults share passive symptoms; paid inspections reveal distinguishing evidence. Compare the learned policy with fixed and adaptive checklists and random actions on fresh, balanced episodes, including higher loads.')
p('<b>Prospective success gate:</b> at least 95% diagnosis accuracy at familiar load and 90% at shifted load, no more than two inspections per episode on average, and no more than 5% unresolved familiar-load cases, across three fresh training seeds. Report per-seed failures and compare costs at matched accuracy; beating expert rules is a separate claim.')
p('Immediate experiment and current evidence','HeadX')
gate_file=root/'reports/evidence/followup/confirmation-particle-gate.json'
confirmation='Three fresh training seeds are being tested before claiming repeatability.'
if gate_file.exists():
    confirmed=json.loads(gate_file.read_text())['methods']['dreamer_particle_actor']['confirmation_gate_passed']
    confirmation=('Three fresh training seeds met the prespecified synthetic gate.' if confirmed else 'Fresh-seed confirmation did not meet every prespecified threshold; further work remains.')
p('The initial agent mostly escalated. Correcting training incentives and diagnostic coverage, then voting across 16 native latent-state/actor samples, achieved 100% development accuracy with one inspection per incident. '+confirmation+' All failed trials are retained. The next scientific gate is transfer to independent dynamics and real lab services.')
p('Path to end-to-end resolution','HeadX')
p('After the synthetic gate, validate on real services confined to the lab namespace. A frozen LLM can then explain the diagnostic evidence and propose a bounded repair. A separate approval/execution layer applies the change, and independent health checks verify recovery. LLM integration and real-service recovery are subsequent gates, not capabilities demonstrated today.')
p('Execution, isolation and deliverables','HeadX')
p('Run bounded jobs on AMD Halo in <b>systems-world-model-lab</b>, capped at four CPUs and 6 GiB RAM, with default-deny networking, no cluster credentials or host mounts, and no changes to other workloads. Retain checkpoints, code, episode-level evidence and a research PDF. Track shared-host health; namespace isolation alone cannot prove zero contention.')
p('Project: <link href="https://github.com/rnishtala/systems-world-model" color="#087E8B">github.com/rnishtala/systems-world-model</link> | Method: <link href="https://danijar.com/project/dreamerv3/" color="#087E8B">DreamerV3</link> | Experimental proposal, not a claim of production readiness.','SmallX')
out=root/'output/pdf/systems-world-model-proposal.pdf';out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=34,bottomMargin=30,title='Systems World Model: One-page Research Proposal',author='Systems World Model project').build(story)
print(out)
