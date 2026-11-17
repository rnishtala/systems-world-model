"""Label-independent intermittent service telemetry loss."""
import hashlib,math
import numpy as np
MASK_SEED=20261117

def mask_sequence(values,columns,incident_id):
    x=np.array(values,copy=True);assert x.shape==(20,2*len(columns))
    services=sorted({c.split('_',1)[0] for c in columns})
    for frame in range(10,20):
        order=sorted(services,key=lambda s:hashlib.sha256(f'{MASK_SEED}:{incident_id}:{frame}:{s}'.encode()).digest())
        hidden=set(order[:math.ceil(len(services)/2)])
        for j,col in enumerate(columns):
            if col.split('_',1)[0] in hidden:x[frame,j]=x[frame,len(columns)+j]=0
    return x

def observation(path,columns,mode):
    from benchmark.dreamer import sequence
    x=sequence(path,columns)
    if mode=='half-services':return mask_sequence(x,columns,path.stem)
    assert mode=='none';return x
