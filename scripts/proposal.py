"""Render the one-page proposal, describing the problem and proposed DreamerV3 triage approach."""
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

p('RESEARCH PROPOSAL | SYSTEMS WORLD MODEL | 11 NOVEMBER 2026', 'SmallX')
p('Learning system behavior for reliable triage', 'TitleX')
p('The problem: one error, several possible causes', 'HeadX')
p('Consider a request flowing through <b>checkout &rarr; payment &rarr; ledger &rarr; PostgreSQL</b>. Customers see slow checkouts, failed payments or timeout errors. The error may appear in checkout even when the underlying problem is in a dependency. Similar symptoms can arise from processing contention in a service, waiting for a database connection, or a delayed downstream call. Multiple faults can occur together.')
p('Triage must determine <b>where the failure originates, which mechanism explains it, and what evidence would distinguish competing explanations</b>. For a payment timeout, we might inspect processing queues, connection-pool wait times and request traces. We want to separate the originating fault from propagated symptoms, recognize combined faults, and avoid collecting every available signal before reaching a diagnosis.')
p('Why a world model could help', 'HeadX')
p('A snapshot of high latency cannot explain how an incident developed. A world model learns how observations evolve over time and how diagnostic actions reveal additional evidence. Its evolving internal state could connect upstream errors with earlier downstream changes, retain evidence across checks, and support choosing the next check that best separates plausible causes.')
p('What DreamerV3 is and how we would use it', 'HeadX')
p('<b>DreamerV3</b> is a model-based reinforcement learning algorithm: it learns a compact model of an environment and trains an action-selection policy using imagined future trajectories. We would reuse its implementation and train a new world model on service telemetry, diagnostic actions and their observed outcomes. The system-specific behavior would be learned from incident sequences, rather than supplied by pretrained game weights.')
d = Drawing(500,85)
for x, title, sub in [(0, 'Incident evidence', 'metrics + request traces'), (127, 'DreamerV3 model', 'learn temporal behavior'), (254, 'Diagnostic policy', 'choose a useful check'), (381, 'Triage outcome', 'faults + evidence')]:
    d.add(Rect(x,31,117,42,rx=4,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue))
    d.add(String(x+7,57,title,fontName='Helvetica-Bold',fontSize=9,fillColor=blue))
    d.add(String(x+7,41,sub,fontName='Helvetica',fontSize=7.5,fillColor=gray))
for x in (127,254,381):
    d.add(Line(x-10,52,x,52,strokeColor=teal))
    d.add(Polygon([x,52,x-4,55,x-4,49],fillColor=teal,strokeColor=teal))
d.add(String(0,13,'Proposed loop: run the selected check, return its real observations, and update the diagnosis.',fontName='Helvetica',fontSize=8,fillColor=gray))
story.append(d)
p('Proposed approach', 'HeadX')
p('<b>1. Learn service behavior.</b> Collect sequences from healthy operation and controlled single and combined faults. Train Dreamer\'s recurrent world model to represent incident history and predict subsequent observations; use known fault labels to train a diagnostic readout.<br/><b>2. Learn efficient investigation.</b> Treat inspections as actions. Use imagined sequences to train a policy rewarded for correct diagnoses and penalized for unnecessary checks. At runtime, execute the selected check and update the model with real evidence.<br/><b>3. Return an evidence-backed diagnosis.</b> Identify the implicated service and fault mechanism, link the conclusion to observed signals, and request further evidence or abstain when the case is outside the model\'s validated coverage.')
p('How we would judge success', 'HeadX')
p('Compare against raw-telemetry classifiers and fixed diagnostic checklists on fresh incidents. Measure accuracy of service and fault identification, combined-fault detection, number of checks and time to diagnosis. Test changed workloads, fault severities and unseen faults. Success means higher accuracy at matched investigation cost, or fewer checks at matched accuracy. Predicted outcomes must be validated against real observations; the benefit of learned dynamics is a research question.')
p('Project: <link href="https://github.com/rnishtala/systems-world-model" color="#087E8B">github.com/rnishtala/systems-world-model</link><br/>Method: <link href="https://danijar.com/project/dreamerv3/" color="#087E8B">DreamerV3 - Mastering Diverse Control Tasks through World Models</link>', 'SmallX')
out = root / 'output/pdf/systems-world-model-proposal.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(out), pagesize=(612,792), leftMargin=48, rightMargin=48, topMargin=30, bottomMargin=28, title='Systems World Model: One-page Research Proposal', author='Systems World Model project').build(story)
print(out)
