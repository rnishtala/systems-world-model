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
p('<b>Objective.</b> Test whether a learned model of service behavior improves triage: distinguish failures with similar symptoms, recognize simultaneous faults, and identify the next useful diagnostic check. The goal is more accurate diagnoses with less investigation effort.')
p('What is DreamerV3?', 'HeadX')
p('DreamerV3 is a model-based reinforcement learning algorithm. It learns a compact, evolving representation of an environment from observations and actions, predicts future outcomes, and trains an action-selection policy using imagined trajectories. Its world model provides memory of what happened and a way to reason about what could happen next.')
p('How we use it for system triage', 'HeadX')
p('We reuse the upstream DreamerV3 implementation and train a new world model from scratch on measured service telemetry and diagnostic actions. Its recurrent state summarizes the incident over time. Today, auxiliary fault supervision helps train that state, and a frozen classifier converts it into a diagnosis. Next, we will test whether imagined diagnostic sequences can teach a policy to select informative checks.')
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
p('What the POC has demonstrated', 'HeadX')
p('A real checkout, payment, ledger and PostgreSQL service chain provides controlled incidents: processing contention, connection-pool contention and downstream delay. Healthy operation, individual faults and their combinations form eight regimes. Training used 160 measured incidents spanning all eight; the diagnosis API is deployed in the isolated lab.')
p('<b>Independent confirmation: 80/80 exact diagnoses.</b> A separately trained model achieved 10/10 per regime. <b>Fresh live API validation: 80/80</b> on a separate batch with no further fitting. These results support known-regime triage using fixed checks. The raw-telemetry baseline scored 78/80 in confirmation; this small difference does not establish that a world model is superior.')
p('Research plan: make triage more effective', 'HeadX')
p('<b>1. Prove the value of temporal state.</b> Compare Dreamer-based diagnosis against raw telemetry, non-temporal representations and fixed checklists. Test whether incident history helps separate similar symptoms and combined faults.<br/><b>2. Learn which evidence to gather.</b> Treat diagnostic checks as actions and reward correct diagnoses while charging for inspection cost. Use Dreamer\'s imagined trajectories to train a diagnostic policy, then validate its choices on fresh real-service incidents.<br/><b>3. Test limits and uncertainty.</b> Evaluate changed workloads, fault severities, startup behavior and service topology. Introduce unknown faults and test when the system should request more evidence or abstain.')
p('<b>Success criteria.</b> On prespecified fresh tests, improve exact diagnosis accuracy at matched inspection cost, or reduce checks and time at matched accuracy. Report results by fault family, across independent training runs, with unknown-fault detection and abstention quality. Current evidence does not yet establish learned investigation planning or generalization to unseen systems.')
p('Project: <link href="https://github.com/rnishtala/systems-world-model" color="#087E8B">github.com/rnishtala/systems-world-model</link> | Method: <link href="https://danijar.com/project/dreamerv3/" color="#087E8B">DreamerV3</link><br/>Evidence: reports/LATENT_CONFIRMATION_RESULTS.md and reports/LATENT_API_RESULTS.md', 'SmallX')
out = root / 'output/pdf/systems-world-model-proposal.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(out), pagesize=(612,792), leftMargin=48, rightMargin=48, topMargin=30, bottomMargin=28, title='Systems World Model: One-page Research Proposal', author='Systems World Model project').build(story)
print(out)
