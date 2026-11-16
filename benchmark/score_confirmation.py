"""Score already-frozen predictions locally; never used inside the cluster job."""
import argparse,hashlib,json
from pathlib import Path
from benchmark.evaluate import score

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--answers',required=True);a=p.parse_args()
    root=Path(a.evidence);hashes=json.loads((root/'frozen-hashes.json').read_text())
    names=[f'{folder}-{seed}/{name}' for seed in (42,43) for folder,name in [('readout','post_mean_full-predictions.json'),('readout','raw-predictions.json'),('baseline','predictions.json')]]
    for name in names:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name],name
    # Read test truth only after complete prediction coverage and hashes are frozen.
    answers=json.loads(Path(a.answers).read_text());truth={r['id']:r for r in answers};test=[r for r in answers if r['split']=='test']
    results={};paired={};preds={}
    for name in names:
        pred=json.loads((root/name).read_text());results[name]=score(pred,answers);preds[name]={r['id']:r['ranks'][0] for r in pred}
    for seed in (42,43):
        dreamer=preds[f'readout-{seed}/post_mean_full-predictions.json']
        for baseline in (f'baseline-{seed}/predictions.json',f'readout-{seed}/raw-predictions.json'):
            other=preds[baseline];counts=dict(both_correct=0,dreamer_only=0,baseline_only=0,both_wrong=0)
            for row in test:
                d=dreamer[row['id']]==row['service'];b=other[row['id']]==row['service']
                key='both_correct' if d and b else 'dreamer_only' if d else 'baseline_only' if b else 'both_wrong';counts[key]+=1
            paired[baseline]=counts
    out=root/'scores.json';assert not out.exists(),'Do not overwrite confirmation scores'
    out.write_text(json.dumps(dict(results=results,paired=paired,test_cases=len(test)),indent=2))
    print(json.dumps({name:dict(n=r['test']['n'],top1=r['test']['ac_at_k']['1'],top3=r['test']['ac_at_k']['3']) for name,r in results.items()},indent=2))
if __name__=='__main__':main()
