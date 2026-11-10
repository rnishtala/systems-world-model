"""Score exact diagnoses and individual mechanisms without conflating the two."""
import json,sys,statistics
from collections import Counter
from microservice.mechanisms import FAULT_MASK,NAMES
rows=[json.loads(l) for l in open(sys.argv[1])]
if not rows:raise ValueError('Empty results')
models={}
for seed in sorted({r['method'] for r in rows}):
    rs=[r for r in rows if r['method']==seed]
    if len({r['episode'] for r in rs})!=len(rs):raise ValueError('Duplicate incident')
    families={}
    for fault in range(8):
        group=[r for r in rs if r['fault']==fault]
        correct=sum(r['diagnosis']==FAULT_MASK[fault] for r in group)
        families[fault]={'n':len(group),'correct':correct,'accuracy':correct/len(group) if group else None}
    mechanisms={}
    for i,name in enumerate(NAMES):
        tp=fp=fn=0
        for r in rs:
            truth=bool(FAULT_MASK[r['fault']]&(1<<i));pred=r['diagnosis']>=0 and bool(r['diagnosis']&(1<<i))
            tp+=truth and pred;fp+=not truth and pred;fn+=truth and not pred
        mechanisms[name]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    models[seed]={'families':families,'mechanisms':mechanisms,'median_seconds':statistics.median(r['seconds'] for r in rs),'escalations':sum(r['diagnosis']==-1 for r in rs),'development_gate':all(v['n']==10 and v['correct']>=(10 if f==0 else 9) for f,v in families.items())}
print(json.dumps({'physical_incidents':len({r['episode'] for r in rows}),'models':models,'passed':all(m['development_gate'] for m in models.values()),'scope':'Single development seed; pairs trained, triple composition held out. Not reliability confirmation.'},indent=2))
