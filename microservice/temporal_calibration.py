"""Incident-held-out screening of current-window diagnosis; no Dreamer training."""
import argparse,json,os,random,math
from pathlib import Path
import numpy as np
from microservice import live
from microservice.temporal_pilot import incident

LABELS=('healthy','database','upstream')
FIELDS=('count','error_rate','latency_p95','cpu_mean','pool_wait_p95','db_mean','db_query_count','db_query_mean','downstream_p95','self_p95')
GAUGES=('active_requests','pool_waiters','pool_in_use')
DELTAS=('original_requests','admission_rejections','downstream_attempts','downstream_timeouts')
FEATURES=[s+'.'+k for s in live.SERVICES for k in FIELDS+GAUGES+DELTAS]

def schedule(seed):
    """Separate collection seeds; nuisance ranges are shared across labels/splits."""
    result=[]
    for split,offset in [('train',0),('test',1)]:
        rng=random.Random(seed+offset)
        cases=[]
        for label in LABELS:
            for _ in range(6):
                profile=dict(healthy_hold=rng.uniform(.03,.05),fault_hold=rng.uniform(.08,.12),base_rate=rng.uniform(18,34),burst_rate=rng.uniform(65,115),onset=rng.uniform(2,4),duration=11,relief_at=30)
                cases.append(dict(split=split,mechanism=label,profile=profile))
        rng.shuffle(cases);result.extend(cases)
    return result

def snapshot_features(windows):
    # Fixed observation time; no event timestamps, controls, IDs or cumulative history.
    i=next(i for i,w in enumerate(windows) if w['elapsed']>=10)
    if i==0:raise ValueError('Missing preceding counter snapshot')
    now,previous=windows[i],windows[i-1];values=[]
    for s in live.SERVICES:
        m=now['metrics'][s];c=m['counters'];old=previous['metrics'][s]['counters']
        values.extend(m[k] for k in FIELDS)
        values.extend(c.get(k,0) for k in GAUGES)
        values.extend(c.get(k,0)-old.get(k,0) for k in DELTAS)
    if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid current-window telemetry')
    return values

def fit_tree(x,y,depth=2):
    """Fixed depth-two CART, minimum two incidents per leaf; train-only thresholds."""
    x=np.asarray(x,float);y=np.asarray(y,int)
    counts=np.bincount(y,minlength=3);node={'prediction':int(np.argmax(counts)),'counts':counts.tolist()}
    def impurity(labels):
        c=np.bincount(labels,minlength=3)
        return len(labels)-float(c@c)/len(labels)
    best=None;best_score=impurity(y)
    if depth<=0 or len(y)<4:return node
    for j in range(x.shape[1]):
        unique=np.unique(x[:,j])
        for t in (unique[:-1]+unique[1:])/2:
            left=x[:,j]<=t
            if min(left.sum(),(~left).sum())<2:continue
            score=impurity(y[left])+impurity(y[~left])
            if score<best_score-1e-12:best_score=score;best=(j,float(t),left)
    if best is None:return node
    j,t,left=best
    node.update(feature=j,threshold=t,left=fit_tree(x[left],y[left],depth-1),right=fit_tree(x[~left],y[~left],depth-1))
    return node

def predict(tree,x):
    while 'feature' in tree:tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
    return tree['prediction']

def evaluate(tree,rows):
    confusion=[[0]*3 for _ in LABELS]
    for r in rows:confusion[r['label']][predict(tree,r['features'])]+=1
    n=len(rows);correct=sum(confusion[i][i] for i in range(3));p=correct/n;z=1.96;d=1+z*z/n
    middle=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(n=n,correct=correct,accuracy=p,wilson95=[middle-half,middle+half],confusion=confusion,
                snapshot_promising=correct>=17 and all(confusion[i][i]>=5 for i in range(3)) and sum(confusion[0][1:])==0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cases=schedule(a.seed)
    protocol=dict(scope='Small randomized snapshot screening; not confirmatory or Dreamer evaluation',seed=a.seed,labels=LABELS,features=FEATURES,cases=cases,classifier='CART depth 2, min leaf 2, deterministic training-only thresholds',diagnosis_seconds=10,screen='At least 17/18 correct, at least 5/6 per class, no healthy false alarms; descriptive only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];tree=None
    for i,case in enumerate(cases):
        result=incident(case['mechanism'],'none',case['profile'])
        (out/f'incident-{i}.json').write_text(json.dumps(result))
        row=dict(split=case['split'],label=LABELS.index(case['mechanism']),features=snapshot_features(result['windows']),restored=result['restored'])
        rows.append(row)
        if i==17:
            tree=fit_tree([r['features'] for r in rows],[r['label'] for r in rows])
            (out/'frozen-classifier.json').write_text(json.dumps(dict(tree=tree,features=FEATURES),indent=2))
        print('SNAPSHOT_CALIBRATION_PROGRESS',i+1,'/36',flush=True)
    (out/'features.json').write_text(json.dumps(rows))
    score=evaluate(tree,[r for r in rows if r['split']=='test'])
    summary=dict(scope=protocol['scope'],incidents=[],calibration=dict(**score,labels=LABELS,features=FEATURES,tree=tree,training_incidents=18,test_incidents=18,all_restored=all(r['restored'] for r in rows)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SNAPSHOT_CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':main()
