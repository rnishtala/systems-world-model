"""Render the auxiliary-supervision study from measured evidence."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/auxiliary'
s=json.loads((E/'summary.json').read_text());m=s['methods'];training=json.loads((E/'training-21.json').read_text())
st=getSampleStyleSheet()
for name,size,leading,color in [('T',20,24,'#123B59'),('H',12,16,'#087E8B'),('B',9.5,12.8,'#182C3A')]:st.add(ParagraphStyle(name=name,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=6))
story=[]
def p(text,style='B'):story.append(Paragraph(text,st[style]))
def table(rows,widths):
 t=Table(rows,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,10))
p('Teaching a world model to retain fault mechanisms','T')
p('Systems World Model | Development experiment | 10 November 2026')
p('Abstract','H')
p(f"An auxiliary-supervised DreamerV3 model diagnosed {m['dreamer_auxiliary']['correct']}/80 fresh incidents exactly, compared with {m['raw']['correct']}/80 for a linear telemetry baseline and {m['dreamer_reward']['correct']}/80 for the same model's imagined-reward diagnosis. The auxiliary head {'passed' if m['dreamer_auxiliary']['gate_passed'] else 'failed'} the prespecified per-family development gates. This is a single-seed experiment, not a reliability confirmation.")
p('Question and architecture','H')
p('Previous frozen-state classifiers failed to generalize reliably to fault combinations. Can an explicit training objective make CPU contention, pool contention and delay independently readable from the Dreamer recurrent state?')
table([['Training path','Inference path'],['Telemetry -> encoder -> RSSM','Telemetry -> encoder -> RSSM'],['RSSM -> reconstruction + reward losses','RSSM -> 3 sigmoid probabilities'],['RSSM -> 3 fault-label losses','Fixed threshold -> fault-set diagnosis'],['Labels enter the loss only','No labels or fault controls']], [247,247])
p('We retain the pinned upstream DreamerV3 encoder, recurrent state-space model (RSSM), decoder and actor-critic training. A new linear three-output head adds binary cross-entropy to the original training objective (weight 1.0). Its gradients update the shared latent representation. This is supervised world-model training; it is neither an LLM fine-tune nor an unmodified Dreamer policy.')
p('Method','H')
p(f"Training uses the original 140 calibration incidents from checkout -> payment -> ledger -> PostgreSQL: 20 each for healthy, three single faults and three pairs. The triple is absent. Seed 21 trained from scratch for {training['steps']:,} total steps; the resumed process took {training['seconds']/60:.1f} minutes (earlier training and connectivity downtime excluded), on CPU within the isolated AMD Halo namespace. A separate 2,000-step smoke checkpoint verified integration before the full run.")
p('Supervision applies only after all three inspections and before terminal diagnosis. Labels are excluded from encoder and decoder construction and stripped before policy computation. At inference, an internal constant placeholder satisfies the replay-space contract; the API accepts telemetry and action costs only. Changing the placeholder from all-zero to all-one leaves policy state and outputs exactly identical under the same random seed.')
p('Inference fixes the CPU, pool and trace inspections, averages each probability over 16 posterior particles, and uses threshold 0.5. These settings were frozen before training. No threshold or model selection used the fresh test batch.')
story.append(PageBreak())
p('Fresh physical-incident evaluation','T')
p('Scenario seed 6501 | Increased load and severity | Ten incidents per family')
rows=[['Family','Auxiliary','Reward','Telemetry']]
for i,name in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','CPU + pool + delay']):
 rows.append([name,*[str(m[k]['families'][str(i)]['correct'])+'/10' for k in ('dreamer_auxiliary','dreamer_reward','raw')]])
rows.append(['Total',*[str(m[k]['correct'])+'/80' for k in ('dreamer_auxiliary','dreamer_reward','raw')]])
table(rows,[245,83,83,83])
p('Gate: all ten healthy incidents correct, and at least nine exact fault sets in every other family. Failed requests or fault-lease overruns invalidate collection. All methods use the same preserved incident trajectories; inference scoring occurs after collection, not as an online resolution trial.')
p(f"Exploratory frozen-state ridge probes scored {m['latent']['correct']}/80 with true training labels and {m['shuffled_latent']['correct']}/80 with shuffled labels. All probes were fitted before collection, using penalty 1 and threshold zero. The telemetry comparator uses eight measured features; the recurrent representation also incorporates observation history and masks. These are not capacity- or information-matched models.")
p('A seed-6500 batch was interrupted by monitoring cleanup before scoring. Its 23 completed records are retained as incomplete; the entire batch was repeated at seed 6501 with unchanged weights, probes and gates.')
p('Interpretation and limits','H')
p('All three primary methods scored 70/70 on healthy, single and pair families. On triples, the auxiliary head omitted delay nine times. Reward diagnosis omitted delay twice and pool once; telemetry omitted pool three times. Equal reward and telemetry totals conceal four disagreements. The model fits all 140 calibration diagnoses.')
p('The auxiliary output is a supervised diagnostic readout of a world model. Success would show that the representation can support this small diagnosis task, not that imagined rollouts outperform classification. The original imagined-reward readout is retained as a comparator. No learned repair action, intervention forecast, LLM integration or autonomous recovery is demonstrated. Triple-fault calibration followed by fresh testing would address known-regime coverage, but would cease to be an unseen-combination test.')
p('This experiment uses one training seed, one small microservice topology and a replay surrogate of measured trajectories. Shared weights previously saw all pair families; only the triple is a new combination. Comparisons with older experiments use different batches and seeds and cannot isolate the effect of supervision. Independent confirmation and an otherwise identical no-supervision run are required before a causal or reliability claim.')
p('Reproducibility and isolation','H')
p('The full checkpoint remained unchanged during evaluation. The existing deployment was restored; no candidate was promoted. Work was confined to the standalone repository and systems-world-model-lab namespace, with an existing 4 CPU / 6 Gi resource cap. Labels remain runner/trainer data, not inference API inputs.')
p('Protocol: reports/AUXILIARY_SUPERVISION_PROTOCOL.md. Evidence: reports/evidence/auxiliary/. Training adapter: microservice/supervision.py. Reproduction: scripts/finish_auxiliary.py. Upstream revision: e01491fad6434b2245a3b8ca201dd7faedcc458c, github.com/danijar/dreamerv3. Technical research note; not peer reviewed.')
def foot(c,d):c.setFont('Helvetica',8);c.drawString(50,25,'Systems World Model | Auxiliary mechanism supervision');c.drawRightString(545,25,str(d.page))
out=R/'output/pdf/dreamerv3-auxiliary-supervision.pdf'
SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=foot,onLaterPages=foot);print(out)
