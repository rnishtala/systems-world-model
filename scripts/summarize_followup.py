"""Keep development thresholds separate from three-seed confirmation."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('files',nargs='+',type=Path);a=p.parse_args()
runs=[json.loads(f.read_text()) for f in a.files]
methods=sorted({r['method'] for run in runs for r in run['rows']})
out={'label':a.label,'training_seeds':[r['training_seed'] for r in runs],'methods':{}}
for method in methods:
    measurements=[]
    for run in runs:
        for shifted in (False,True):
            rows=[r for r in run['rows'] if r['method']==method and r['shifted']==shifted]
            if not rows:raise ValueError('Missing evaluation regime')
            if len({r['episode'] for r in rows}) != len(rows):
                raise ValueError('Duplicate episode identifiers')
            if any(r['correct'] != (r['diagnosis'] == r['fault']) for r in rows):
                raise ValueError('Correctness flag disagrees with diagnosis and ground truth')
            if len(rows)==150 and any(sum(r['fault']==f for r in rows)!=50 for f in (0,1,2)):
                raise ValueError('Expected balanced fault classes')
            if any(not 0 <= r['checks'] <= r['steps'] <= 6 for r in rows):
                raise ValueError('Invalid bounded episode accounting')
            if len(runs)==3 and set(out['training_seeds'])=={10,11,12}:
                expected_seed=4000000+run['training_seed']+100000*shifted
                if any(r['eval_seed']!=expected_seed for r in rows):
                    raise ValueError('Confirmation evaluation stream differs from frozen protocol')
            m=dict(training_seed=run['training_seed'],shifted=shifted,episodes=len(rows),
                   accuracy=statistics.mean(r['correct'] for r in rows),
                   checks=statistics.mean(r['checks'] for r in rows),
                   unresolved=statistics.mean(r['diagnosis']==-1 for r in rows))
            m['meets_thresholds']=(len(rows)==150 and m['accuracy']>=(.9 if shifted else .95)
                                   and m['checks']<=2 and (shifted or m['unresolved']<=.05))
            measurements.append(m)
    passes=all(m['meets_thresholds'] for m in measurements)
    out['methods'][method]={'per_seed':measurements,'development_meets_thresholds':passes,
        'confirmation_gate_passed':passes and len(runs)==3 and set(out['training_seeds'])=={10,11,12}}
print(json.dumps(out,indent=2))
