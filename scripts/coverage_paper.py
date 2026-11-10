"""Research note generated from the completed coverage experiment."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/complete-coverage'
completion=json.loads((E/'completion.json').read_text());seeds=[22,23] if completion['confirmation_passed'] is not None else [22]
results={seed:json.loads((E/f'eval-{seed}/summary.json').read_text())['methods'] for seed in seeds}
st=getSampleStyleSheet()
for name,size,leading,color in [('T',20,24,'#123B59'),('H',12,16,'#087E8B'),('B',9.5,12.8,'#182C3A')]:st.add(ParagraphStyle(name=name,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=7))
story=[]
def p(text,style='B'):story.append(Paragraph(text,st[style]))
def table(rows,widths):
 t=Table(rows,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
p('Dreamer triage with complete fault-regime coverage','T')
p('Systems World Model | Isolated AMD Halo experiment | 10 November 2026')
p('Abstract','H')
m=results[22]
p(f"After adding newly collected triple-fault calibration, the primary Dreamer reward readout scored {m['dreamer_reward']['correct']}/80 exact diagnoses on fresh incidents; telemetry alone scored {m['raw']['correct']}/80. The prespecified development gate {'passed' if completion['development_passed'] else 'failed'}. "+('An independently trained seed was tested on a second fresh batch; its gate '+('passed.' if completion['confirmation_passed'] else 'failed.') if 23 in seeds else 'Independent confirmation was skipped as prespecified after the primary development failure.'))
p('Question and scope','H')
p('The preceding model correctly diagnosed all 70 healthy, single-fault and pair-fault test incidents, but only seven of ten unseen triples through its reward readout. This study asks whether explicit coverage of every known combination improves reliable triage of new incidents. The triple is now represented in training: no unseen-combination claim is made.')
p('System and training','H')
p('Checkout -> payment -> ledger -> PostgreSQL produces real HTTP/SQL telemetry. The runner independently controls CPU contention, connection-pool contention and downstream delay. The agent receives requested telemetry only; it cannot access fault controls or labels through the inference API.')
table([['Training','Inference'],['160 measured calibration incidents','Fresh incident, fixed checkpoint'],['Encoder -> recurrent world model','Inspect CPU, pool and trace'],['Original Dreamer losses + 3 label losses','16 posterior-state particles'],['Labels excluded from encoder and decoder','Imagine eight diagnosis rewards -> argmax']], [247,247])
p('The original 140 calibration incidents are retained unchanged. Twenty new triple-fault incidents (seed 6600, alternating ordinary and increased severity) produce 20 examples per family across all eight regimes. Previous held-out incidents, including interrupted runs, are excluded. A hash of the original calibration is checked before augmentation.')
p('Pinned upstream DreamerV3 e01491f, size1m, CPU float32 and 100,000 steps are retained. Three independent sigmoid fault targets add binary cross-entropy with weight 1 after complete inspection and before terminal diagnosis. This trains the shared representation. The main decision rule remains imagined-reward argmax; the auxiliary head and ridge telemetry classifier are comparators. All rules were fixed before new collection.')
p('Sequential protocol','H')
p('Train development seed 22 from scratch and evaluate 80 fresh incidents at scenario seed 6601: ten per family at increased load/severity. Only if the primary readout passes every gate, train independent seed 23 on the same calibration and evaluate a different 80-incident batch at seed 6602. Gate: 10/10 healthy and at least 9/10 exact fault sets in every other family. No test-driven threshold changes or candidate promotion.')
story.append(PageBreak())
p('Measured results','T')
for seed in seeds:
 m=results[seed];t=json.loads((E/f'training-{seed}.json').read_text())
 p(f"{'Development' if seed==22 else 'Independent confirmation'}: model seed {seed}",'H')
 p(f"Training: {t['seconds']/60:.2f} minutes; 100,000 steps. Fresh physical incidents scored offline from preserved trajectories.")
 rows=[['Family','Reward','Auxiliary','Telemetry']]
 for i,name in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']):rows.append([name,*[str(m[k]['families'][str(i)]['correct'])+'/10' for k in ('dreamer_reward','dreamer_auxiliary','raw')]])
 rows.append(['Total',*[str(m[k]['correct'])+'/80' for k in ('dreamer_reward','dreamer_auxiliary','raw')]])
 table(rows,[245,83,83,83])
p('Error analysis and next step','H')
p('All primary methods classified the first healthy incident as delay. Its cumulative ledger p95 remained about 440 ms although measured batches took only 60-70 ms. Counts include the two warm-up batches; snapshots retain them until control reset. Warm-up latency therefore persisted in measured observations. The specific startup cause is not established, and the failed row remains in the score. All three missed pool in one triple; the reward readout also missed pool in one CPU-plus-pool case.')
p('Next: establish a consistent measurement window excluding warm-up samples while preserving fault state and connections. Evaluate startup transients separately and investigate weak pool-pressure signals. Freeze that contract before fresh validation; do not tune thresholds or discard errors from this batch.')
p('Interpretation and limits','H')
p('Passing this small-sample gate would support a bounded demonstration across known regimes, not production reliability, arbitrary root-cause discovery or statistical proof of 90% accuracy in every family. The training surrogate replays measured trajectories; the study does not demonstrate learned repair dynamics, autonomous remediation or LLM integration. A world model matching telemetry alone does not establish added value from imagined planning.')
p('No model or threshold is adjusted against these fresh batches. Earlier experiments used different seeds and batches, so cross-study score changes do not isolate the effect of triple coverage. A matched no-auxiliary ablation and broader topologies remain separate research work. Full calibration fit is not evidence of generalization.')
p('Audit and reproducibility','H')
p('Audits exclude labels from encoder and decoder and verify identical policy outputs for opposite label placeholders. Checkpoint digests remain unchanged during evaluation. Requests and incident durations are checked; fault leases and runner reset bound fault injection. The preserved default deployment is restored after the experiment. All cluster work stays within systems-world-model-lab and its 4 CPU / 6 Gi budget.')
p('Protocol: reports/COMPLETE_COVERAGE_PROTOCOL.md. Evidence: reports/evidence/complete-coverage/. Execution: scripts/finish_coverage.py. Upstream: github.com/danijar/dreamerv3, revision e01491fad6434b2245a3b8ca201dd7faedcc458c. Technical research note; not peer reviewed.')
def foot(c,d):c.setFont('Helvetica',8);c.drawString(50,25,'Systems World Model | Complete regime coverage');c.drawRightString(545,25,str(d.page))
out=R/'output/pdf/dreamerv3-complete-coverage.pdf';SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=foot,onLaterPages=foot);print(out)
