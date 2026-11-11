"""Render the one-page proposal, separating validated progress from future work."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

root = Path(__file__).resolve().parents[1]
blue = colors.HexColor('#123B59')
teal = colors.HexColor('#087E8B')
gray = colors.HexColor('#536271')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=blue, spaceAfter=9))
styles.add(ParagraphStyle(name='BodyX', fontName='Helvetica', fontSize=10, leading=13.5, spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX', fontName='Helvetica', fontSize=8, leading=11, textColor=gray, spaceAfter=7))
styles.add(ParagraphStyle(name='HeadX', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=teal, spaceBefore=6, spaceAfter=4))
story = []
def p(text, style='BodyX'):
    story.append(Paragraph(text, styles[style]))

p('RESEARCH PROPOSAL | SYSTEMS WORLD MODEL | 10 NOVEMBER 2026', 'SmallX')
p('Learning system behavior for reliable triage', 'TitleX')
p('<b>Objective.</b> Learn how a service system behaves over time, use that representation to distinguish interacting failures, and test whether it can guide investigation and safe recovery. The first milestone is a deployed, diagnosis-only proof of concept; adaptive investigation and LLM-assisted resolution are the next research steps.')
p('Architecture: what is working today', 'HeadX')
p('A real checkout, payment, ledger and PostgreSQL service chain supplies measured telemetry. We reuse upstream DreamerV3 and train a new recurrent world model from scratch, with auxiliary fault supervision. A frozen classifier translates its learned state into a fault set. This is a separately trained model, not LLM fine-tuning or reuse of pretrained game weights.')
d = Drawing(500,78)
for x, title, sub in [(0, 'Service telemetry', 'fixed diagnostic checks'), (127, 'DreamerV3 state', 'temporal representation'), (254, 'Frozen classifier', 'identify combined faults'), (381, 'Diagnosis API', 'return the fault set')]:
    d.add(Rect(x,24,117,42,rx=4,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue))
    d.add(String(x+7,50,title,fontName='Helvetica-Bold',fontSize=9,fillColor=blue))
    d.add(String(x+7,34,sub,fontName='Helvetica',fontSize=7.5,fillColor=gray))
for x in (127,254,381):
    d.add(Line(x-10,45,x,45,strokeColor=teal))
    d.add(Polygon([x,45,x-4,48,x-4,42],fillColor=teal,strokeColor=teal))
d.add(String(0,7,'Current decisions use learned state + classifier; imagined rewards do not select the diagnosis.',fontName='Helvetica',fontSize=8,fillColor=gray))
story.append(d)
p('Completed milestone: real-service triage', 'HeadX')
p('Controlled incidents cover processing contention, database connection-pool contention and downstream delay: healthy operation, each single fault, every pair and all three together. The model was calibrated on 160 measured incidents spanning these eight known regimes. The inference API is now the isolated lab default; rollback and post-deployment health checks passed.')
p('<b>Independent confirmation: 80/80 exact diagnoses.</b> A separately trained model passed with 10/10 in every regime. <b>Fresh live API validation: 80/80</b> on a separate batch, without further training or fitting. Saved-trajectory replay also matched 80/80; that verifies serving parity, not additional accuracy. Live fault-setup-through-diagnosis time was 5.81 seconds median and 8.96 seconds at the 95th percentile, including warm-up and measurement.')
p('What these results establish - and what remains open', 'HeadX')
p('The POC supports triage of known fault combinations in one small service chain using fixed inspections. All eight regimes were represented in training. The confirmation raw-telemetry baseline scored 78/80; this small difference does not establish world-model superiority. Unseen systems, unknown faults, startup behavior and production reliability remain untested. Learned investigation planning and autonomous repair are not demonstrated.')
p('Next research: from diagnosis to verified resolution', 'HeadX')
p('<b>1. Test generalization.</b> Freeze the model and evaluate changed workloads, fault severities and service topology; measure unknown-fault abstention and failures.<br/><b>2. Establish the model\'s contribution.</b> Compare with raw telemetry, non-temporal representations and checklists at matched accuracy and inspection cost.<br/><b>3. Add an LLM resolution loop.</b> Let an LLM explain evidence and propose diagnostic checks or bounded repairs. Test world-model predictions before execution; retain approval, independent recovery checks and rollback.')
p('<b>Success criteria.</b> Prespecify fresh-test gates, report exact diagnosis accuracy by fault family, inspection cost, latency and recovery success, and retain failed trials. Keep experiments in the project\'s isolated namespace. Deliver reproducible code, frozen artifacts, incident-level evidence and an updated research paper.')
p('Project: <link href="https://github.com/rnishtala/systems-world-model" color="#087E8B">github.com/rnishtala/systems-world-model</link> | Method: <link href="https://danijar.com/project/dreamerv3/" color="#087E8B">DreamerV3</link><br/>Evidence: reports/LATENT_CONFIRMATION_RESULTS.md and reports/LATENT_API_RESULTS.md', 'SmallX')
out = root / 'output/pdf/systems-world-model-proposal.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(out), pagesize=(612,792), leftMargin=48, rightMargin=48, topMargin=30, bottomMargin=28, title='Systems World Model: One-page Research Proposal', author='Systems World Model project').build(story)
print(out)
