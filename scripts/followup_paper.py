"""Generate the follow-up report only after all prespecified confirmation runs."""
import json,statistics
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'reports/evidence/followup'
seeds=(10,11,12)
training=[json.loads((E/f'confirmation/results-{s}.json').read_text()) for s in seeds]
particles=[json.loads((E/f'confirmation/particle-results-{s}.json').read_text()) for s in seeds]
execution=json.loads((E/'confirmation-execution.json').read_text())
impact=json.loads((E/'confirmation-cluster-comparison.json').read_text())
for seed,run,result in zip(seeds,training,particles):
    assert run['training_seed']==result['training_seed']==seed
    assert run['steps']==100000 and run['training_abstention_penalty']==2 and run['inspection_coverage']==.5
    for shift in (False,True):
        rows=[r for r in result['rows'] if r['method']=='dreamer_particle_actor' and r['shifted']==shift]
        assert len(rows)==150 and all(r['eval_seed']==4000000+seed+100000*shift for r in rows)

def measure(runs,method,shift=False):
    rows=[r for run in runs for r in run['rows'] if r['method']==method and r['shifted']==shift]
    return dict(n=len(rows),accuracy=statistics.mean(r['correct'] for r in rows),checks=statistics.mean(r['checks'] for r in rows),unresolved=statistics.mean(r['diagnosis']==-1 for r in rows),reward=statistics.mean(r['reward'] for r in rows))
def pct(x):return f'{100*x:.1f}%'
def thresholds(x,shift=False):return x['accuracy']>=(.9 if shift else .95) and x['checks']<=2 and (shift or x['unresolved']<=.05)
gate=all(thresholds(measure([run],'dreamer_particle_actor',shift),shift) for run in particles for shift in (False,True))
familiar=measure(particles,'dreamer_particle_actor');shifted=measure(particles,'dreamer_particle_actor',True)
blue=colors.HexColor('#123B59');teal=colors.HexColor('#087E8B');gray=colors.HexColor('#536271')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=blue,spaceAfter=12));styles.add(ParagraphStyle(name='BodyX',fontSize=9.5,leading=13,spaceAfter=8));styles.add(ParagraphStyle(name='SmallX',fontSize=8,leading=10.5,textColor=gray,spaceAfter=7));styles.add(ParagraphStyle(name='HeadX',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=blue,spaceBefore=9,spaceAfter=7))
story=[]
def p(s,style='BodyX'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'HeadX')
def table(rows,widths):
    t=Table([[Paragraph(escape(str(v)),styles['SmallX']) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,blue),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')])]))
    story.append(t);story.append(Spacer(1,7))

p('SYSTEMS WORLD MODEL / FOLLOW-UP EXPERIMENT','SmallX')
p('From abstention to<br/>model-based diagnosis','TitleX')
p('DreamerV3 adaptation, controlled failures, and fresh-seed confirmation','HeadX')
p('Technical research report | 9 November 2026 | Synthetic POC | Not peer reviewed','SmallX')
h('Abstract')
p(f'We investigate whether DreamerV3 can learn diagnostic behavior in a bounded partially observed queue simulator. An initial configuration largely escalated rather than inspecting. Changing abstention cost alone produced guessing; supplying inspection experience alone retained abstention. A combined 100,000-step training configuration learned useful predictions and substantially improved the native actor. Aggregating 16 native latent-state/actor particles reached 100% accuracy in a development evaluation. We froze this condition and trained three fresh seeds. Confirmation accuracy is {pct(familiar["accuracy"])} at familiar load and {pct(shifted["accuracy"])} at shifted load, with {familiar["checks"]:.2f} and {shifted["checks"]:.2f} inspections per episode respectively. The prespecified per-seed gate '+('passed.' if gate else 'did not pass.')+' All development failures and confirmation outcomes are retained. The study demonstrates only what these synthetic measurements support; real-system transfer and LLM-assisted repair remain untested.')
h('Question and contribution')
p('Can a learned world model support evidence-gathering diagnosis when faults share passive symptoms? The contribution is a reproducible adaptation and an audit of its learning and inference failures, not a new general-purpose reinforcement-learning algorithm. We reuse the project-linked DreamerV3 implementation [1, 2], train new weights, and evaluate actual diagnostic actions.')
h('Architecture')
d=Drawing(490,105)
for x,w,title,sub in [(0,150,'Queue observations','real simulator feedback'),(170,150,'DreamerV3 x 16','latent states + native actor'),(340,150,'Action vote','inspect / diagnose / escalate')]:
    d.add(Rect(x,42,w,46,rx=5,fillColor=colors.HexColor('#E4EFF3'),strokeColor=blue));d.add(String(x+8,69,title,fontName='Helvetica-Bold',fontSize=10,fillColor=blue));d.add(String(x+8,52,sub,fontName='Helvetica',fontSize=8,fillColor=gray))
for x in (170,340):d.add(Line(x-20,65,x,65,strokeColor=teal));d.add(Polygon([x,65,x-5,68,x-5,62],fillColor=teal,strokeColor=teal))
d.add(String(0,21,'All particles observe the same incident and receive the same executed action.',fontName='Helvetica',fontSize=9,fillColor=gray));story.append(d)
p('Figure 1. Sixteen particles are inference states of one trained model, not sixteen separately trained models. No LLM, expert diagnosis rule, or production telemetry is used.','SmallX')
story.append(PageBreak())
h('1. Environment and training method')
p('The environment has three classes: healthy operation, CPU bottleneck and downstream slowdown. CPU and downstream faults share passive transition distributions. A scalar queue evolves with load and capacity; observations include queue, derived latency/errors, two inspected sensor values, visibility flags and time. Ground truth is used for reward and scoring, never as an inference feature.')
table([['Element','Frozen confirmation configuration'],['Observations / actions','Nine numeric observations; wait, two inspections, three diagnoses, escalation. At most six actions per episode.'],['Original evaluation reward','-0.05 per action; -0.15 additional inspection cost; +2 correct / -2 wrong diagnosis; -0.5 escalation or timeout.'],['Training change','Escalation/timeout penalty 2.0. Evaluation retains the original 0.5 penalty. Other rewards and physical transitions are unchanged.'],['Inspection coverage','50% of training episodes: inspect both sensors in random order, then choose a uniformly random diagnosis. No hidden labels or expert answer is provided to the controller.'],['Learning budget','100,000 upstream driver steps per seed, including reset transitions. Train from scratch; replay capacity 10,000; batch 8 x 16; training ratio 16.'],['Model / implementation','638,671 parameters; upstream size1m configuration; float32 CPU; JAX/JAXlib 0.4.35; Python 3.11.']],[105,385])
p('The upstream encoder, recurrent state-space model (RSSM), observation/reward/continuation heads, replay and imagined actor-critic training are reused. Upstream files are unmodified. A training-only adapter supplies exploratory actions and ensures recurrent previous-action state matches the action actually executed. Otherwise the learned policy controls the episode.')
h('2. Particle inference')
p('At evaluation, 16 native RSSM/actor states receive identical observations. Each produces a sampled action; the most-voted action is executed once. Every particle then receives that same executed action. No extra observations, hidden labels, expert fault thresholds or action restrictions are added. The checkpoint weights are unchanged. This inference wrapper aggregates both latent-state and actor sampling; it does not establish which source of variation explains every error.')
p('Pinned upstream revision: e01491fad6434b2245a3b8ca201dd7faedcc458c. The upstream describes itself as a reimplementation, not a Google-supported software product. Exact source bundles, resolved dependencies, configurations and episode records are retained [3].','SmallX')
story.append(PageBreak())
h('3. Development results and failure analysis')
rows=[['Condition','Steps / seed','Familiar','Shifted','Checks (familiar)']]
base=json.loads((ROOT/'reports/evidence/main-summary.json').read_text())
bf=next(g for g in base['groups'] if g['method']=='dreamer' and not g['shifted']);bs=next(g for g in base['groups'] if g['method']=='dreamer' and g['shifted'])
rows.append(['Initial (3 seeds)','50,000',pct(bf['accuracy']),pct(bs['accuracy']),f'{bf["checks"]:.2f}'])
for label,filename,method,steps in [('Penalty only','penalty-v2/results-0.json','dreamer',50000),('Coverage only','coverage-v3a/results-0.json','dreamer',50000),('Combined, sampled','combined-v4/results-0.json','dreamer',100000),('Combined, actor argmax','combined-v4/greedy-results.json','dreamer_greedy_actor',100000),('Combined, lookahead','combined-v4/planner-results.json','world_model_planner',100000),('Combined, 16 particles','combined-v4/particle-results.json','dreamer_particle_actor',100000)]:
    run=json.loads((E/filename).read_text());x=measure([run],method);y=measure([run],method,True)
    rows.append([label,f'{steps:,}',pct(x['accuracy']),pct(y['accuracy']),f'{x["checks"]:.2f}'])
table(rows,[145,80,70,70,125])
p('Table 1. Follow-ups use one development training seed and 150 balanced episodes per load regime. Inference comparisons reuse the combined checkpoint; they do not each incur new training. Evaluation RNG seeds differ, so these are distribution-matched, not exact paired trajectory comparisons.','SmallX')
p('The initial agent learned a cheap abstention strategy. A larger penalty removed abstention but did not teach inspection. Coverage alone exposed diagnostic evidence but left the actor abstaining. The combined run changed both training conditions and doubled the budget; its improvement cannot be attributed to any one of these changes.')
h('Frozen-model probe')
probe1=json.loads((E/'coverage-v3a/probe.json').read_text());probe2=json.loads((E/'combined-v4/probe.json').read_text())
p(f'A separate probe forced both inspections in 90 episodes. Reward ranking by the world model identified the correct class in {pct(probe1["world_model_reward_ranking_accuracy_after_checks"])} after coverage-only training and {pct(probe2["world_model_reward_ranking_accuracy_after_checks"])} after combined training. The combined model\'s one-step diagnostic reward RMSE was {probe2["imagination_reward_rmse"]:.3f}. These are conditional prediction results, not autonomous diagnosis accuracy. They helped distinguish prediction learning from action-selection failure.')
p('A two-step lookahead controller and actor argmax did not close the development gap. Particle voting did. This supports testing stochastic inference aggregation, but a single selected development checkpoint is insufficient evidence of reliability. Two adapter-launch failures were retained as engineering logs and excluded from performance results because they produced no valid evaluations.')
story.append(PageBreak())
h('4. Fresh-seed confirmation')
p('After development, training and the 16-particle controller were frozen. Seeds 10, 11 and 12 trained from scratch. Each was evaluated on 150 balanced episodes at familiar load (0.55-0.85) and 150 at shifted load (0.85-0.98), using previously untouched evaluation seeds. No configuration was tuned against these confirmation outcomes.')
rows=[['Seed','Familiar accuracy','Shifted accuracy','Checks (F / S)','Unresolved (F)']]
for seed,run in zip(seeds,particles):
    x=measure([run],'dreamer_particle_actor');y=measure([run],'dreamer_particle_actor',True)
    rows.append([seed,pct(x['accuracy']),pct(y['accuracy']),f'{x["checks"]:.2f} / {y["checks"]:.2f}',pct(x['unresolved'])])
rows.append(['Aggregate',pct(familiar['accuracy']),pct(shifted['accuracy']),f'{familiar["checks"]:.2f} / {shifted["checks"]:.2f}',pct(familiar['unresolved'])]);table(rows,[70,100,100,115,105])
p('Table 2. 450 episodes per regime across three independently trained models. Abstentions/timeouts count as unresolved, not correct. The per-seed values, rather than only the aggregate, determine the gate.','SmallX')
p('Prespecified gate: each seed must reach at least 95% familiar-load accuracy and 90% shifted-load accuracy, no more than two inspections per episode on average in either regime, and no more than 5% unresolved familiar-load incidents. <b>Outcome: '+('PASS.' if gate else 'NOT PASSED.')+'</b> This is an empirical POC criterion, not a statistical guarantee of production reliability.')
rows=[['Confirmation comparator','Familiar accuracy','Shifted accuracy','Checks (F)']]
for label,runs,method in [('Native sampled actor',training,'dreamer'),('16-particle actor',particles,'dreamer_particle_actor'),('Adaptive checklist',particles,'adaptive'),('Fixed checklist',particles,'checklist'),('Random',particles,'random')]:
    x=measure(runs,method);y=measure(runs,method,True);rows.append([label,pct(x['accuracy']),pct(y['accuracy']),f'{x["checks"]:.2f}'])
table(rows,[160,110,110,110])
p('Table 3. The adaptive checklist does not exploit all information in the passive queue trajectory. Fewer inspections than this comparator does not establish superiority to an optimal expert rule. Particle voting also adds inference compute, which inspection counts alone do not capture.','SmallX')
starts=[datetime.fromisoformat(j['startTime'].replace('Z','+00:00')) for j in execution['jobs']];ends=[datetime.fromisoformat(j['completionTime'].replace('Z','+00:00')) for j in execution['jobs']]
elapsed=(max(ends)-min(starts)).total_seconds()/60
p(f'The three confirmation jobs completed in {elapsed:.1f} minutes of elapsed batch time. Individual training wall times were '+', '.join(f'{r["train_seconds"]/60:.1f}' for r in training)+' minutes and overlapped. Wall-clock durations are not CPU-time measurements.')
story.append(PageBreak())
h('5. AMD Halo execution and isolation')
p('Experiments ran on an AMD Ryzen AI MAX+ 395 Halo with 16 physical cores / 32 logical CPUs and about 125 GiB RAM, using CPU only. All workloads, source ConfigMaps and storage belong to systems-world-model-lab. This experiment did not modify workloads outside its namespace. The namespace retained a four-CPU and 6 GiB aggregate limit throughout; the pod-count quota changed from two to three for concurrent confirmation.')
p('Each confirmation job had a 1333m CPU / 2 GiB memory limit and its own source, replay, logs and checkpoints. All three shared the original aggregate resource budget. Containers ran non-root with dropped capabilities, read-only root filesystems, no Kubernetes credentials and no host mounts. Default-deny network policy remained active; follow-ups reused installed dependencies without reopening download access.')
p(f'Snapshot comparison recorded {impact["before_running"]} and {impact["after_running"]} running pods outside the lab, {len(impact["readiness_regressions"])} readiness regressions; the count of existing pods with increased restarts was {len(impact["restart_increases"])}. The Argo CD controller already had repeated OOM terminations before the initial experiment. Readiness comparisons apply to matching pod identities; concurrent outside-lab pod replacements are listed separately in the evidence. Snapshot checks cannot prove zero latency impact or exclude transient/shared-host contention. Full context is retained with the evidence.')
p('The PVC requests 10 GiB through local-path storage; this request is not a hard per-directory disk quota. Completed jobs use no running training processes. Checkpoints are retained for reproducibility, and temporary collectors are removed after export.')
h('6. Limits and next gate')
p('This is a small synthetic task with three familiar classes, strongly separable sensors, a scalar queue, short episodes and no concurrent faults. Load shift does not test novel faults or service topologies. Guided training improves observation coverage but adds a domain-specific exploration schedule. Sixteen particles increase inference compute, and only three confirmation models are available. No calibrated causal probabilities or real-system transfer are claimed.')
p('Next, keep the selected configuration frozen and evaluate independent simulator dynamics and bounded real services in the lab namespace. Introduce overlapping sensors, missing data, delayed effects and unseen faults. Only after transfer is demonstrated should a frozen LLM explain the evidence and propose a reviewed repair, with separate execution controls and independent recovery verification. No LLM or real repair was evaluated here.')
h('References and reproducibility')
p('[1] Hafner et al. Mastering Diverse Control Tasks through World Models. https://danijar.com/project/dreamerv3/','SmallX')
p('[2] Project-linked implementation: https://github.com/danijar/dreamerv3 . Revision: e01491fad6434b2245a3b8ca201dd7faedcc458c.','SmallX')
p('[3] Code, protocol, failed trials, exact dependencies, source hashes, episode-level evidence and report generator: https://github.com/rnishtala/systems-world-model . See reports/FOLLOWUP_PROTOCOL.md and reports/evidence/followup/.','SmallX')
out=ROOT/'output/pdf/dreamerv3-triage-followup.pdf';out.parent.mkdir(parents=True,exist_ok=True)
def footer(c,doc):
    c.setStrokeColor(teal);c.line(48,39,564,39);c.setFont('Helvetica',8);c.setFillColor(gray);c.drawString(48,26,'SYSTEMS WORLD MODEL | Synthetic research POC | 2026-11-09');c.drawRightString(564,26,str(doc.page))
SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=54,title='From Abstention to Model-Based Diagnosis: DreamerV3 Follow-up',author='Systems World Model project').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
