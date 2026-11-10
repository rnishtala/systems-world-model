"""Train-only model selection for a frozen world model's diagnostic head."""
import numpy as np

def fit_transform(x,rank):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    z=(x-mean)/scale
    if rank==0:components=np.eye(x.shape[1])
    else:components=np.linalg.svd(z,full_matrices=False)[2][:min(rank,len(x)-1)].T
    return mean,scale,components,z@components

def fit_head(x,y,rank,penalty):
    mean,scale,components,z=fit_transform(np.asarray(x,np.float64),rank)
    bias=y.mean(0);reg=len(x)*penalty
    if z.shape[1]>len(z):w=z.T@np.linalg.solve(z@z.T+reg*np.eye(len(z)),y-bias)
    else:w=np.linalg.solve(z.T@z+reg*np.eye(z.shape[1]),z.T@(y-bias))
    return dict(mean=mean,scale=scale,components=components,weights=w,bias=bias)

def predict(head,x):
    scores=((x-head['mean'])/head['scale'])@head['components']@head['weights']+head['bias']
    return np.sum((scores>0)*[1,2,4],axis=-1)

def folds(target):
    rng=np.random.default_rng(7100);split=np.empty(len(target),int)
    for label in sorted(set(target)):
        indices=np.flatnonzero(target==label);rng.shuffle(indices);split[indices]=np.arange(len(indices))%5
    result=[(split!=k,split==k,'stratified') for k in range(5)]
    result += [(target!=mask,target==mask,'held_pair') for mask in (3,5,6)]
    return result

def select(x,y,ranks):
    target=np.sum((y>0)*[1,2,4],axis=1);splits=folds(target);records=[]
    for rank in ranks:
        prepared=[]
        for train,test,kind in splits:
            mean,scale,components,z=fit_transform(np.asarray(x[train],np.float64),rank)
            prepared.append((z,((x[test]-mean)/scale)@components,y[train],target[test],kind))
        for penalty in (.01,.1,1.,10.,100.):
            scores={'stratified':[0,0],'held_pair':[0,0]}
            for z,zt,yt,truth,kind in prepared:
                bias=yt.mean(0);reg=len(z)*penalty
                if z.shape[1]>len(z):w=z.T@np.linalg.solve(z@z.T+reg*np.eye(len(z)),yt-bias)
                else:w=np.linalg.solve(z.T@z+reg*np.eye(z.shape[1]),z.T@(yt-bias))
                pred=np.sum(((zt@w+bias)>0)*[1,2,4],axis=1)
                scores[kind][0]+=int(sum(pred==truth));scores[kind][1]+=len(truth)
            score=sum(a/b for a,b in scores.values())/2
            records.append(dict(rank=rank,penalty=penalty,score=score,counts=scores))
    best=max(records,key=lambda r:(r['score'],-(r['rank'] or x.shape[1]),r['penalty']))
    return fit_head(x,y,best['rank'],best['penalty']),{'chosen':best,'candidates':records,'scope':'Head selection only; frozen Dreamer previously trained on all calibration incidents.'}

def hybrid_transform(x,rank,weight):
    # Last eight coordinates are raw telemetry; remaining coordinates are latent.
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1;z=(x-mean)/scale
    if weight==0:return mean,scale,np.eye(x.shape[1])[:,-8:],z[:,-8:]
    rank=min(rank,len(z)-1,z.shape[1]-8)
    pc=np.linalg.svd(z[:,:-8],full_matrices=False)[2][:rank].T
    projection=z[:,:-8]@pc;pc=pc/np.maximum(projection.std(0),1e-6)*weight
    components=np.zeros((x.shape[1],rank+8));components[:-8,:rank]=pc;components[-8:,rank:]=np.eye(8)
    return mean,scale,components,z@components

def select_hybrid(latent,raw,y):
    x=np.concatenate([latent,raw],axis=1).astype(np.float64)
    target=np.sum((y>0)*[1,2,4],axis=1);splits=folds(target);records=[]
    for rank,weight in [(0,0)]+[(r,w) for r in (8,16,32) for w in (.1,1.)]:
        prepared=[]
        for train,test,kind in splits:
            mean,scale,c,z=hybrid_transform(x[train],rank,weight)
            prepared.append((z,((x[test]-mean)/scale)@c,y[train],target[test],kind))
        for penalty in (.01,.1,1.,10.,100.):
            scores={'stratified':[0,0],'held_pair':[0,0]}
            for z,zt,yt,truth,kind in prepared:
                bias=yt.mean(0);w=np.linalg.solve(z.T@z+len(z)*penalty*np.eye(z.shape[1]),z.T@(yt-bias))
                pred=np.sum(((zt@w+bias)>0)*[1,2,4],axis=1)
                scores[kind][0]+=int(sum(pred==truth));scores[kind][1]+=len(truth)
            records.append(dict(rank=rank,latent_weight=weight,penalty=penalty,score=sum(a/b for a,b in scores.values())/2,counts=scores))
    best=max(records,key=lambda r:(r['score'],-r['rank'],-r['latent_weight'],r['penalty']))
    mean,scale,c,z=hybrid_transform(x,best['rank'],best['latent_weight']);bias=y.mean(0)
    w=np.linalg.solve(z.T@z+len(z)*best['penalty']*np.eye(z.shape[1]),z.T@(y-bias))
    return dict(mean=mean,scale=scale,components=c,weights=w,bias=bias),{'chosen':best,'candidates':records}
