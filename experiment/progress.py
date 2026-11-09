"""Read-only compact status for bounded experiment outputs."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[10,11,12]);a=p.parse_args()
for seed in a.seeds:
    metrics=a.root/f'train-{seed}/metrics.jsonl';latest={}
    if metrics.exists():
        for line in reversed(metrics.read_text().splitlines()):
            try:latest=json.loads(line);break
            except json.JSONDecodeError:pass
    status={'seed':seed,'step':latest.get('step',0),'steps_per_second':latest.get('fps/policy')}
    result=a.root/f'results-{seed}.json';particle=a.root/f'particle-results-{seed}.json'
    status['phase']='training'
    if result.exists():status.update(phase='particle evaluation',step=json.loads(result.read_text())['steps'])
    if particle.exists():
        d=json.loads(particle.read_text());status['phase']='complete'
        status['accuracy']={str(shift):sum(r['correct'] for r in d['rows'] if r['method']=='dreamer_particle_actor' and r['shifted']==shift)/150 for shift in (False,True)}
    print(json.dumps(status),flush=True)
