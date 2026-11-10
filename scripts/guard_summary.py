"""Score fresh guarded pilot; earlier confirmation is calibration, not validation."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
sets={n:[json.loads(x) for x in (root/(n+'.jsonl')).read_text().splitlines()] for n in ('known','cpu-delay','cpu-pool','pool-delay')}
out={'scope':'Fresh pilot of empirical guard with fixed previously trained weights. Not general unknown-fault reliability.','physical_incidents':sum(len({r['episode'] for r in rows}) for rows in sets.values()),'seeds':{}}
for seed in (10,11,12):
 known=[r for r in sets['known'] if r['method']==seed]
 classes={str(f):{'n':sum(r['fault']==f for r in known),'accuracy':statistics.mean(r['diagnosis']==f for r in known if r['fault']==f)} for f in range(4)}
 combined={}
 for name in ('cpu-delay','cpu-pool','pool-delay'):
  rows=[r for r in sets[name] if r['method']==seed]
  combined[name]={'n':len(rows),'escalation_rate':statistics.mean(r['diagnosis']==-1 for r in rows),'incomplete_known_answer_rate':statistics.mean(r['diagnosis']>=0 for r in rows)}
 allrows=[r for rows in sets.values() for r in rows if r['method']==seed]
 record={'underlying_candidate_accuracy':statistics.mean(max(range(4),key=lambda i:r['predicted_diagnosis_rewards'][i])==r['fault'] for r in known),'known_classes':classes,'known_false_escalation_rate':statistics.mean(r['diagnosis']==-1 for r in known),'healthy_false_alarm_rate':statistics.mean(r['diagnosis']>0 for r in known if r['fault']==0),'combined':combined,'median_seconds':statistics.median(r['seconds'] for r in allrows),'max_checks':max(r['checks'] for r in allrows)}
 record['rejected_correct_candidates']=sum(r['diagnosis']==-1 and max(range(4),key=lambda i:r['predicted_diagnosis_rewards'][i])==r['fault'] for r in known)
 record['rejected_incorrect_candidates']=sum(r['diagnosis']==-1 and max(range(4),key=lambda i:r['predicted_diagnosis_rewards'][i])!=r['fault'] for r in known)
 record['passed']=all(classes[str(f)]['n']==15 and classes[str(f)]['accuracy']>=(.95 if f==0 else .9) for f in range(4)) and record['known_false_escalation_rate']<=.05 and all(v['n']==12 and v['escalation_rate']>=.9 for v in combined.values()) and record['median_seconds']<60 and record['max_checks']<=3
 out['seeds'][str(seed)]=record
out['passed']=all(v['passed'] for v in out['seeds'].values())
print(json.dumps(out,indent=2))
