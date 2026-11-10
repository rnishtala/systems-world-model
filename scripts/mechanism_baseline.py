"""Independent depth-three binary trees, fit only on calibration observations.
Offline comparator uses the same fixed inspection order as Dreamer's harness.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from microservice.environment import Incident
from microservice.mechanisms import FAULT_MASK
FEATURES=list(range(2,10));MAX_DEPTH=3;MIN_LEAF=5

def fit(x,y,depth=0):
    positive=int(y.sum());node={'prediction':int(positive>len(y)/2),'n':len(y),'positive':positive}
    if depth==MAX_DEPTH or positive in (0,len(y)):return node
    best=None
    for feature in FEATURES:
        values=np.unique(x[:,feature])
        for threshold in (values[:-1]+values[1:])/2:
            left=x[:,feature]<=threshold
            if min(left.sum(),(~left).sum())<MIN_LEAF:continue
            score=sum(len(g)*(g.mean()*(1-g.mean())) for g in (y[left],y[~left]))
            if best is None or score<best[0]:best=(score,feature,float(threshold),left)
    parent=len(y)*y.mean()*(1-y.mean())
    if best is None or best[0]>=parent:return node
    _,feature,threshold,left=best
    return {**node,'feature':feature,'threshold':threshold,'left':fit(x[left],y[left],depth+1),'right':fit(x[~left],y[~left],depth+1)}

def predict(tree,x):
    while 'feature' in tree:tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
    return tree['prediction']

def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--model',required=True);p.add_argument('--live');a=p.parse_args()
    model=Path(a.model)
    if a.live:
        trees=json.loads(model.read_text())['trees'];rows=[]
        for r in map(json.loads,Path(a.live).read_text().splitlines()):
            x=r['trace'][-1]['observation'];start=time.perf_counter();mask=sum(predict(t,x)<<i for i,t in enumerate(trees))
            rows.append({'episode':r['episode'],'fault':r['fault'],'source_method':r['method'],'method':'binary-trees','diagnosis':mask,'correct':mask==FAULT_MASK[r['fault']],'seconds':time.perf_counter()-start})
        print(json.dumps({'rows':rows,'scope':'Same live observations, offline classification timing only; not end-to-end latency.'}))
    else:
        if model.exists():raise ValueError('Refusing to replace fitted baseline')
        rows=list(map(json.loads,Path(a.calibration).read_text().splitlines()));xs=[]
        for r in rows:
            env=Incident(r['frames'],r['fault'],compositional=True)
            for act in (1,2,3):x,_,_=env.step(act)
            xs.append(x)
        x=np.array(xs);masks=np.array([FAULT_MASK[r['fault']] for r in rows])
        import hashlib
        model.write_text(json.dumps({'algorithm':'independent binary CART, depth 3, min leaf 5, majority ties absent','calibration_sha256':hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),'features':FEATURES,'trees':[fit(x,((masks>>i)&1)) for i in range(3)]},indent=2)+'\n')
if __name__=='__main__':main()
