"""Diagnostic readouts, labels and immutable artifact hashes."""
import hashlib
from pathlib import Path
import numpy as np
FAULT_MASK={0:0,1:1,2:2,3:4,4:5,5:3,6:6,7:7}

CANDIDATES=[dict(kind='linear',penalty=float(p)) for p in (1000,100,10,1)]+[
    dict(kind='rbf',gamma=g,penalty=p) for g in (.1,1.) for p in (.01,.1)]


def kernel(a,b,gamma):
    d=np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1]
    return np.exp(-gamma*d)


def fit(x,y,config):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale;bias=y.mean(0)
    gram=z@z.T if config['kind']=='linear' else kernel(z,z,config['gamma'])
    dual=np.linalg.solve(gram+config['penalty']*np.eye(len(x)),y-bias)
    return dict(config=config,mean=mean,scale=scale,bias=bias,z=z,dual=dual)


def score(model,x):
    z=(np.asarray(x)-model['mean'])/model['scale'];c=model['config']
    k=z@model['z'].T if c['kind']=='linear' else kernel(z,model['z'],c['gamma'])
    return k@model['dual']+model['bias']


def diagnose(values):return np.sum((values>0)*np.array([1,2,4]),axis=-1)


def choose(x,y,groups):
    target=diagnose(y);counts=[]
    for config in CANDIDATES:
        pred=np.empty(len(y),int)
        for group in sorted(set(groups)):
            held=groups==group;train=~held
            pred[held]=diagnose(score(fit(x[train],y[train],config),x[held]))
        counts.append(int(sum(pred==target)))
    return CANDIDATES[int(np.argmax(counts))],counts

def labels(rows):
    masks=np.array([FAULT_MASK[r['fault']] for r in rows])
    return ((masks[:,None]>>np.arange(3))&1)*2-1


def digest(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def stats(pred,rows):
    truth=diagnose(labels(rows));families=np.array([r['fault'] for r in rows])
    return dict(n=len(rows),correct=int(sum(pred==truth)),families={str(f):dict(n=int(sum(families==f)),correct=int(sum((pred==truth)&(families==f)))) for f in range(8)})
