"""Check this refactor against archived numerical evidence; no cluster access."""
import argparse,ast,importlib.util,json
from pathlib import Path
import numpy as np
from microservice.probe_target import vector
from microservice.readout import choose,fit,labels
from microservice.validate import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();root=a.evidence
    train=[json.loads(l) for l in (root/'collection/train.jsonl').read_text().splitlines()]
    test=[json.loads(l) for l in (root/'evaluation/test.jsonl').read_text().splitlines()]
    for name,rows in [('train',train),('test',test)]:
        with np.load(root/f'evaluation/{name}-features.npz') as xs:
            for t in (5,7):np.testing.assert_array_equal(xs[f'history_{t}'],[np.concatenate([vector(r,j) for j in range(t+1)]) for r in rows])
    frozen=json.loads((root/'evaluation/frozen.json').read_text());y=labels(train);groups=np.array([r['repeat']%4 for r in train])
    with np.load(root/'evaluation/train-features.npz') as xs,np.load(root/'evaluation/readouts.npz') as saved:
        for name,c in frozen['configs'].items():
            x=xs[f'{name[:-1]}_{int(name[-1])-1}'];config,counts=choose(x,y,groups)
            assert dict(config=config,cv_correct=counts)==c
            model=fit(x,y,config)
            for k,v in model.items():
                if k!='config':np.testing.assert_allclose(v,saved[f'{name}__{k}'],rtol=1e-8,atol=1e-9)
    pred=json.loads((root/'evaluation/predictions.json').read_text())['methods'];actual=summarize({n:np.array(v) for n,v in pred.items()},test)
    old=json.loads((root/'evaluation/summary.json').read_text())
    for key,value in actual.items():assert old[key]==value,key
    print('PASS: observation vectors, training-only selection, fitted coefficients, scores and gates match archived evidence')
if __name__=='__main__':main()
