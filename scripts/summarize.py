"""Aggregate episode evidence without hiding poor seeds or shifted-load failures."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
files=sorted(a.directory.glob('results-*.json'))
runs=[json.loads(f.read_text()) for f in files]
summary={'runs':len(runs),'training_seconds':[r['train_seconds'] for r in runs],'groups':[]}
for shift in (False,True):
    for method in ('dreamer','checklist','adaptive','random'):
        rows=[x for r in runs for x in r['rows'] if x['shifted']==shift and x['method']==method]
        if not rows:continue
        seed_accuracy=[statistics.mean(x['correct'] for x in r['rows'] if x['shifted']==shift and x['method']==method) for r in runs]
        summary['groups'].append(dict(method=method,shifted=shift,episodes=len(rows),
            accuracy=statistics.mean(x['correct'] for x in rows),
            accuracy_by_training_seed=seed_accuracy,
            checks=statistics.mean(x['checks'] for x in rows),
            reward=statistics.mean(x['reward'] for x in rows),
            escalation=statistics.mean(x['diagnosis']==-1 for x in rows)))
print(json.dumps(summary,indent=2))
