"""One local scoring pass after all partial-telemetry predictions are frozen."""
import argparse,hashlib,json
from pathlib import Path
from benchmark.evaluate import score

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--answers',required=True);a=p.parse_args();root=Path(a.evidence)
    hashes=json.loads((root/'frozen-hashes.json').read_text())
    names=[f'readout-{variant}-{seed}-{condition}/{method}-predictions.json' for variant in ('world','control') for seed in (42,43) for condition in ('partial','full') for method in ('post_mean_full','raw')]
    predictions={}
    for name in names:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name],name
        predictions[name]=json.loads((root/name).read_text())
    for seed in (42,43):
        for condition in ('partial','full'):
            assert predictions[f'readout-world-{seed}-{condition}/raw-predictions.json']==predictions[f'readout-control-{seed}-{condition}/raw-predictions.json'],'Raw comparators must be identical'
    # This is the only point where held-out answers enter the scoring process.
    answers=json.loads(Path(a.answers).read_text());test=[r for r in answers if r['split']=='test'];results={k:score(v,answers) for k,v in predictions.items()}
    paired={};errors={};gate={}
    for condition in ('partial','full'):
        for seed in (42,43):
            key=f'{seed}-{condition}';world={r['id']:r['ranks'][0] for r in predictions[f'readout-world-{seed}-{condition}/post_mean_full-predictions.json']}
            errors[key]=[dict(id=r['id'],service=r['service'],fault=r['fault'],predicted=world[r['id']]) for r in test if world[r['id']]!=r['service']]
            for comparator,name in [('control',f'readout-control-{seed}-{condition}/post_mean_full-predictions.json'),('raw',f'readout-world-{seed}-{condition}/raw-predictions.json')]:
                other={r['id']:r['ranks'][0] for r in predictions[name]};counts=dict(both_correct=0,world_only=0,comparator_only=0,both_wrong=0)
                for r in test:
                    w=world[r['id']]==r['service'];b=other[r['id']]==r['service']
                    counts['both_correct' if w and b else 'world_only' if w else 'comparator_only' if b else 'both_wrong']+=1
                paired[key+'-'+comparator]=counts
                if condition=='partial':gate[key+'-'+comparator]=(counts['world_only']-counts['comparator_only'])/len(test)>=.10-1e-9
    out=root/'scores.json';assert not out.exists(),'Do not overwrite frozen scores'
    out.write_text(json.dumps(dict(test_cases=len(test),results=results,paired=paired,world_errors=errors,primary_gate_checks=gate,primary_gate_passed=all(gate.values())),indent=2))
    print(json.dumps(dict(primary_gate_passed=all(gate.values()),test_scores={k:dict(top1=v['test']['ac_at_k']['1'],top3=v['test']['ac_at_k']['3']) for k,v in results.items()}),indent=2))
if __name__=='__main__':main()
