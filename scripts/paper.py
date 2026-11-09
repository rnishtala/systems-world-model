"""Generate a research-style report from measured episode evidence only."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'reports/evidence'
main=json.loads((E/'main-summary.json').read_text())
pilot=json.loads((E/'pilot-summary.json').read_text())
impact=json.loads((E/'cluster-comparison.json').read_text())
assert main['runs'] == 3 and pilot['runs'] == 3, 'All prespecified seeds are required'
assert all(g['episodes'] == 450 for g in main['groups']), 'Incomplete evaluation evidence'
blue=colors.HexColor('#123B59');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#536271')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=blue,spaceAfter=14))
styles.add(ParagraphStyle(name='Deck',fontSize=12,leading=17,textColor=gray,spaceAfter=14))
styles.add(ParagraphStyle(name='BodyX',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallX',fontSize=8,leading=11,spaceAfter=6,textColor=gray))
styles['Heading1'].textColor=blue;styles['Heading1'].fontSize=16;styles['Heading1'].leading=20
styles['Heading2'].textColor=teal;styles['Heading2'].fontSize=12
story=[]
def p(s,style='BodyX'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'Heading1')
def table(rows,widths):
    formatted=[[Paragraph(escape(str(c)),styles['SmallX']) for c in row] for row in rows]
    t=Table(formatted,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,blue),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')])]))
    story.append(t);story.append(Spacer(1,10))
def get(summary,method,shift=False):return next(g for g in summary['groups'] if g['method']==method and g['shifted']==shift)
def pct(x):return f'{100*x:.1f}%'
d=get(main,'dreamer');a=get(main,'adaptive');pd=get(pilot,'dreamer')

p('SYSTEMS WORLD MODEL / EXPERIMENT 001','SmallX')
p('DreamerV3 for active\nsystem diagnosis'.replace('\n','<br/>'),'TitleX')
p('A bounded, CPU-only proof of concept on AMD Halo','Deck')
p('Technical research report | 9 November 2026 | Not peer reviewed','SmallX')
h('Abstract')
p(f'We adapt the project-linked DreamerV3 implementation to a small partially observed queue simulator. An agent chooses diagnostic inspections and then identifies healthy operation, a CPU bottleneck, or downstream slowdown. Three independent training seeds use 50,000 environment steps each. On 450 balanced evaluation episodes at familiar load, Dreamer achieves {pct(d["accuracy"])} accuracy with {d["checks"]:.2f} inspections per episode; a hand-designed adaptive checklist achieves {pct(a["accuracy"])} with {a["checks"]:.2f} inspections. A 4,000-step pilot achieved {pct(pd["accuracy"])} accuracy. We report all seeds, shifted-load performance, and deployment constraints. These measurements establish a runnable integration; they do not establish real-system transfer or superiority to expert rules.')
h('Research question')
p('Can learned latent dynamics and a policy trained through imagination select useful inspections when two system faults share the same passive symptoms? The immediate objective is diagnosis, not generating or executing a repair.')
h('Architecture')
drawing=Drawing(480,100)
for x,w,title,sub in [(0,145,'Queue simulator','hidden fault + workload'),(165,145,'DreamerV3','world model + actor-critic'),(330,145,'Inspection policy','inspect / diagnose / escalate')]:
    drawing.add(Rect(x,40,w,48,rx=5,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue))
    drawing.add(String(x+10,69,title,fontName='Helvetica-Bold',fontSize=10,fillColor=blue))
    drawing.add(String(x+10,51,sub,fontSize=8,fillColor=gray))
drawing.add(Line(145,64,165,64,strokeColor=teal));drawing.add(Line(310,64,330,64,strokeColor=teal))
for x in (165,330):drawing.add(Polygon([x,64,x-5,67,x-5,61],fillColor=teal,strokeColor=teal))
drawing.add(String(0,15,'Training: observations + rewards -> latent dynamics -> imagined policy learning',fontSize=9,fillColor=gray))
story.append(drawing)
p('Figure 1. No LLM, production telemetry, or live fault injection is used in this experiment. The simulator executes inside the isolated Kubernetes training job.','SmallX')
story.append(PageBreak())
h('1. Environment and experimental method')
p('A scalar queue evolves with offered load, effective service capacity, and small Gaussian noise. Faulty capacity is 0.4; healthy capacity is 1.0. Latency and error observations are derived from queue growth. CPU and downstream faults have identical passive transition distributions by construction, making an inspection necessary to distinguish them.')
table([['Element','Specification'],['Observations','Nine numeric values: workload, queue, latency, errors, two inspected metrics, two visibility flags, and episode time.'],['Faults','Healthy, CPU bottleneck, downstream slowdown. Fault identity is hidden from the policy.'],['Actions','Wait; inspect CPU; inspect dependency; diagnose one of three classes; escalate.'],['Inspection sensor','Noisy bounded reading; CPU fault produces high CPU, downstream fault produces high dependency latency.'],['Reward','-0.05 per action; -0.15 additional inspection cost; +2 correct / -2 incorrect diagnosis; -0.5 escalation or timeout.'],['Episode bound','At most six actions; outcomes are recorded even when wrong or escalated.']],[92,388])
p('An environment regression test confirms that matched CPU/downstream fault episodes have identical passive observations and pre-diagnosis rewards. Hidden labels are used for reward and evaluation, not inference features. This is a controlled diagnosis benchmark, not a validated physical model of a production service.')
h('2. Upstream reuse and training')
p('The unmodified upstream encoder, recurrent state-space model (RSSM), observation/reward/continuation heads, replay, and actor-critic machinery are reused through a replacement environment factory. The 638,671-parameter configuration uses numeric observations, float32 CPU execution, batch size 8, sequence length 16, and training ratio 16. Weights are learned from scratch. Step budgets use the upstream driver counter, including reset transitions; they are not gradient-update counts.')
p('Upstream revision: e01491fad6434b2245a3b8ca201dd7faedcc458c. JAX/JAXlib 0.4.35, Python 3.11. Complete dependency resolution and per-run configurations accompany the repository evidence. The linked upstream repository describes itself as a reimplementation, not a Google-supported product.')
p('The pilot used three 4,000-step runs. Its weak result motivated one larger, fixed-budget experiment: three fresh 50,000-step runs with fresh evaluation seeds. No reward or architecture changes were selected using the main evaluation results. Small-scale adaptation is not a reproduction of Dreamer benchmark claims.')
story.append(PageBreak())
h('3. Evaluation and results')
p('Each checkpoint is evaluated on 150 balanced episodes at familiar load (0.55-0.85) and 150 at shifted load (0.85-0.98). All methods have the same sensor/action interface. A fixed checklist inspects both metrics; the adaptive checklist stops after high CPU or otherwise inspects downstream. Random chooses uniformly among seven actions. The upstream stochastic policy is sampled at evaluation time. Simulator randomness is distribution-matched, not exactly paired across methods.')
rows=[['Method','Familiar accuracy','Checks','Shifted accuracy','Checks']]
for method in ('dreamer','checklist','adaptive','random'):
    x=get(main,method);y=get(main,method,True)
    rows.append([method.title(),pct(x['accuracy']),f'{x["checks"]:.2f}',pct(y['accuracy']),f'{y["checks"]:.2f}'])
table(rows,[100,100,60,115,105])
p('Table 1. Main experiment: 450 episodes per method per load regime. Abstentions/timeouts count as unresolved, not correct. Fewer checks alone is not an improvement when accuracy falls or escalation rises.','SmallX')
rows=[['Training seed','Familiar accuracy','Shifted accuracy','Train wall time']]
for i,(x,y,t) in enumerate(zip(d['accuracy_by_training_seed'],get(main,'dreamer',True)['accuracy_by_training_seed'],main['training_seconds'])):
    rows.append([i,pct(x),pct(y),f'{t/60:.2f} min'])
table(rows,[100,120,130,130])
p('Table 2. Individual seeds are shown rather than interpreting hundreds of episodes as hundreds of independently trained models. Three seeds are insufficient for a strong population-level performance claim.','SmallX')
p(f'Main Dreamer escalation/timeout rate is {pct(d["escalation"])} at familiar load and {pct(get(main,"dreamer",True)["escalation"])} at shifted load. Mean familiar-load return is {d["reward"]:.3f}, versus {a["reward"]:.3f} for the adaptive checklist. Total main training wall time is {sum(main["training_seconds"])/60:.2f} minutes; setup and evaluation time are excluded.')
p(f'The pilot familiar-load accuracy was {pct(pd["accuracy"])} (seed values: '+', '.join(pct(v) for v in pd['accuracy_by_training_seed'])+'). Pilot and main evidence are both retained. Evaluation loads are new, but fault families and simulator equations are not.')
h('Interpretation')
if d['accuracy']<a['accuracy']-.02:
    p('Under this budget, learned triage does not beat the adaptive rule baseline on accuracy. The experiment supports implementation feasibility, not replacement of expert diagnosis. Additional training, richer partially observed environments, and stronger generalization tests would be needed before claiming operational value. Escalation offers a predictable -0.55 return; blind balanced diagnosis has expected return about -0.72. A policy can therefore improve over blind guessing by abstaining, without discovering the positive-return inspection sequence. This is a plausible optimization explanation, not an ablation-proven cause.')
else:
    p('The learned policy approaches the adaptive rule baseline on this small task. Because the environment is simple and the rules know its fault mechanisms, this is evidence of task learning, not a demonstrated advantage in production triage. Claims about efficiency must also account for training compute and all failed diagnoses.')
story.append(PageBreak())
h('4. Isolation and execution on AMD Halo')
p('Training ran in systems-world-model-lab on the existing AMD Halo (Ryzen AI MAX+ 395, 16 physical cores / 32 logical CPUs, about 125 GiB host RAM), using CPU only. The namespace caps CPU at 4 cores, memory at 6 GiB, and requested persistent storage at 10 GiB. Jobs are sequential, have a two-hour deadline, run non-root with dropped capabilities and a read-only root filesystem, and mount no Kubernetes service-account token or host paths.')
p('Networking starts default-deny. A temporary public-HTTPS and cluster-DNS allowance installs dependencies; it is removed before the main experiment. Probes from the lab could not connect to the cluster API, shared database, or demo service. These observations are reported as blocked-or-unreachable rather than definitive proof of every possible network path.')
p(f'Before/after inspection found {impact["before_running"]} and {impact["after_running"]} running pods outside the lab; {len(impact["readiness_regressions"])} readiness regressions. Existing pods with increased restart counts: {len(impact["restart_increases"])}. Missing pods: {len(impact["missing_pods"])}; new pods: {len(impact["new_pods"])}. No existing workload specification was modified by the experiment. The Argo CD application controller had 1,748 restarts and an OOMKilled termination before deployment; OOM restarts continued during the run under its existing 384 MiB memory limit. This pre-existing instability is recorded, not attributed conclusively to or excluded from experimental effects.')
p('Namespace isolation does not eliminate shared CPU, memory bandwidth, disk, or network contention. Readiness/restart checks cannot prove zero application latency impact. The experiment makes no such claim. Models and datasets remain in the lab PVC; live experiment processes are stopped after evidence collection.')
h('5. Threats to validity and next experiment')
p('The task has only three familiar classes, strongly separable inspection sensors, scalar queue dynamics, short episodes, and no concurrent faults. Higher offered load is a narrow distribution shift, not an unseen service topology. We did not evaluate calibrated causal probabilities, open-loop multi-step prediction on real telemetry, novel-fault recognition, an LLM, code repair, or real-service recovery. The model can exploit simulator regularities even when labels are hidden.')
p('First, isolate the learning failure with a prespecified ablation of abstention reward and inspection exploration, using fresh evaluation seeds. More training alone is not assumed to solve it. Then evaluate frozen checkpoints against a richer independent simulator and real, bounded lab service measurements. Add diagnostics with overlapping likelihoods, delayed/noisy sensors, unseen faults and observation failures. Only after diagnostic benefit is demonstrated should a frozen LLM translate evidence into a reviewed repair proposal, with actual post-deployment recovery verification.')
h('References and reproducibility')
p('[1] Hafner et al. Mastering Diverse Control Tasks through World Models. DreamerV3 project: https://danijar.com/project/dreamerv3/','SmallX')
p('[2] Project-linked implementation: https://github.com/danijar/dreamerv3 (revision recorded above).','SmallX')
p('[3] Ha and Schmidhuber. World Models (2018). https://worldmodels.github.io/','SmallX')
p('Code, episode-level results, training logs, dependency lock, and manifests: https://github.com/rnishtala/systems-world-model . This report is generated from reports/evidence; it is a POC technical report, not a peer-reviewed publication.','SmallX')

out=ROOT/'output/pdf/dreamerv3-system-triage-poc.pdf';out.parent.mkdir(parents=True,exist_ok=True)
def footer(c,doc):
    c.setStrokeColor(teal);c.line(48,40,564,40)
    c.setFont('Helvetica',8);c.setFillColor(gray)
    c.drawString(48,27,'SYSTEMS WORLD MODEL | Reproducible POC report | 2026-11-09')
    c.drawRightString(564,27,str(doc.page))
SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=55,
    title='DreamerV3 for Active System Diagnosis: A Bounded POC',author='Systems World Model project').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
