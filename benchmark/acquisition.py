"""Observation-only diagnostic-check task and training-data-only controls."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from benchmark.dreamer import sequence


def groups(columns,services):
    n=len(columns)
    return [np.array([j for j,c in enumerate(columns) if c.split('_',1)[0]==s]+[n+j for j,c in enumerate(columns) if c.split('_',1)[0]==s],int) for s in services]


def aggregate(path,columns):return sequence(path,columns)[10:].mean(0)


class ObservationOracle:
    def __init__(self,full,group_indices):
        self._full=np.asarray(full);self.groups=group_indices;self.seen=np.zeros(len(group_indices),bool)
        self.visible=np.zeros(len(full),np.float32)
    def observe(self):return np.concatenate([self.visible,self.seen.astype(np.float32)])
    def reveal(self,action):
        if not 0<=action<len(self.seen) or self.seen[action]:raise ValueError('Check must be available and distinct')
        ix=self.groups[action];self.visible[ix]=self._full[ix];self.seen[action]=True
        return self.observe()


def load_training(data,answers):
    rows=json.loads((data/'manifest.json').read_text());assert all(r['split'] in ('train','test') for r in rows)
    train=[r for r in rows if r['split']=='train'];truth={r['id']:r for r in json.loads(answers.read_text())}
    assert set(truth)=={r['id'] for r in train} and all(r['split']=='train' for r in truth.values())
    services=sorted({r['service'] for r in truth.values()});assert len(services)==5
    columns=set()
    for r in train:
        with np.load(data/(r['id']+'.npz')) as d:columns.update(c for c in d['columns'].tolist() if c.split('_',1)[0] in services)
    columns=sorted(columns);group_indices=groups(columns,services);assert all(len(ix) for ix in group_indices)
    x=np.array([aggregate(data/(r['id']+'.npz'),columns) for r in train]);assert np.isfinite(x).all()
    y=np.array([services.index(truth[r['id']]['service']) for r in train])
    return rows,train,services,columns,group_indices,x,y


def fit_shared(x,y,group_indices):
    inputs=[];targets=[];classes=len(group_indices)
    for row,target in zip(x,y):
        for subset in itertools.product((False,True),repeat=classes):
            oracle=ObservationOracle(row,group_indices)
            for action,seen in enumerate(subset):
                if seen:oracle.reveal(action)
            inputs.append(oracle.observe());targets.append(target)
    inputs=np.asarray(inputs,np.float64);mean=inputs.mean(0);scale=inputs.std(0);scale[scale<1e-6]=1
    z=(inputs-mean)/scale;labels=np.eye(classes)[targets];bias=labels.mean(0)
    weights=np.linalg.solve(z.T@z+100*np.eye(z.shape[1]),z.T@(labels-bias))
    return dict(mean=mean,scale=scale,weights=weights,bias=bias)


def rank(probe,obs):
    scores=(obs-probe['mean'])/probe['scale']@probe['weights']+probe['bias'];assert np.isfinite(scores).all()
    return np.argsort(-scores,kind='stable').tolist()


def softmax(x):
    x=x-x.max(-1,keepdims=True);p=np.exp(x);return p/p.sum(-1,keepdims=True)


def retrieval_action(obs,seen,x,y,group_indices):
    # Only revealed coordinates of obs are read. Future outcomes come from TRAIN rows.
    scale=x.std(0);scale[scale<1e-6]=1
    observed=np.concatenate([ix for ix,known in zip(group_indices,seen) if known]) if any(seen) else np.array([],int)
    cost=(((x[:,observed]-obs[observed])/scale[observed])**2).sum(1)
    prior=softmax((-cost/max(len(observed),1))[None])[0];labels=np.eye(len(group_indices))[y]
    values=np.full(len(group_indices),np.inf)
    for action,ix in enumerate(group_indices):
        if seen[action]:continue
        pair=(((x[:,None,ix]-x[None,:,ix])/scale[ix])**2).sum(-1)
        weights=softmax(-(pair+cost[None])/(len(observed)+len(ix)))
        probs=weights@labels;entropy=-(probs*np.log(np.maximum(probs,1e-12))).sum(-1)
        values[action]=prior@entropy
    return int(values.argmin())


def random_order(incident_id,replicate):
    seed=int.from_bytes(hashlib.sha256(f'1901:{incident_id}:{replicate}'.encode()).digest()[:8],'little')
    return np.random.default_rng(seed).permutation(5).tolist()
