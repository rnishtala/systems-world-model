"""One-page report of training-only selection; no new live accuracy claim."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/regularized-head';s=json.loads((E/'selection.json').read_text());h=json.loads((E/'hybrid-selection.json').read_text())
st=getSampleStyleSheet();st.add(ParagraphStyle(name='T',fontSize=21,leading=25,spaceAfter=12,textColor=colors.HexColor('#123B59')));st.add(ParagraphStyle(name='B',fontSize=10,leading=14,spaceAfter=8));st.add(ParagraphStyle(name='H',fontSize=12,leading=16,spaceBefore=6,spaceAfter=6,textColor=colors.HexColor('#087E8B')));story=[]
def p(t,k='B'):story.append(Paragraph(t,st[k]))
p('Regularization did not fix composition','T')
p('Systems World Model | Training-only diagnostic-head selection | 10 November 2026')
p('Result','H')
p('The selected latent classifier scored 120/140 in stratified validation but 0/60 when a complete pair family was withheld from head fitting. All 25 pure-latent settings scored 0/60 on that check. Adding direct telemetry selected zero latent weight. The candidate was rejected before fresh live testing; Dreamer weights and the deployed controller remain unchanged.')
rows=[['Selected head','Stratified','Withheld pair']]
for name,m in [('Latent: rank 16',s['latent']),('Raw telemetry',s['raw']),('Hybrid: zero latent weight',h)]:
 c=m['chosen']['counts'];rows.append([name,*[str(c[k][0])+'/'+str(c[k][1]) for k in ('stratified','held_pair')]])
t=Table(rows,colWidths=[270,110,110]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EFF3')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,8))
p('The 60 stress predictions reuse the original 140 calibration incidents; these are not 200 independent physical incidents. No seed-6500 data were collected. The hybrid winner is the raw classifier, not a world-model improvement.')
p('Method','H')
p('Standardization and PCA were fitted separately inside each training fold. Five stratified folds were supplemented by three leave-one-pair-family-out splits. Candidate latent ranks were 8, 16, 32, 64 and no PCA; averaged-loss ridge penalties were .01, .1, 1, 10 and 100. Threshold zero was fixed. The selection objective averaged the two exact-set accuracies, with deterministic ties.')
p('After the pure-latent grid failed, a documented training-only amendment tested a telemetry skip connection with 8/16/32 latent components at weights .1/1, plus a zero-weight control. No prior held-out data entered either search. Dreamer previously trained on all calibration incidents: only the diagnostic head withheld these pair families.')
p('Interpretation and next experiment','H')
p('These linear heads do not provide reliable compositional triage from the current state representation. This does not prove that nonlinear decoding is impossible or that all fault information is absent. The previous probe recovered some secondary faults. Expanded validation searches are model selection, not unbiased performance estimates.')
p('The next hypothesis is explicit, independent CPU/pool/delay supervision during world-model training, alongside forecasting. Labels must be training targets only, never inference inputs. Fresh validation and a telemetry-only ablation must establish whether the world model adds value. This training change is planned, not implemented here.')
p('Validation and reproducibility','H')
p('Five targeted tests passed, and the hybrid candidate grid reproduced exactly. No cluster operations or Dreamer retraining occurred. Code and artifacts remain in the standalone repo. Protocol: reports/REGULARIZED_HEAD_PROTOCOL.md. Evidence: reports/evidence/regularized-head/. Reproduction: scripts/select_regularized_head.py. Technical research note; not peer reviewed.')
def foot(c,d):c.setFont('Helvetica',8);c.drawString(50,25,'Systems World Model | Regularized head selection');c.drawRightString(545,25,str(d.page))
out=R/'output/pdf/dreamerv3-regularized-head.pdf';SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(story,onFirstPage=foot,onLaterPages=foot);print(out)
