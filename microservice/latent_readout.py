"""Inference-only, checkpoint-bound ridge readout. No fitting or labels here."""
import hashlib,json
from pathlib import Path
import numpy as np

def checkpoint_digest(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def state_features(dyn):
    parts=[]
    for key in ('deter','stoch'):
        value=np.asarray(dyn[key])
        if value.shape[0]!=16:raise ValueError('Expected 16 state particles')
        parts.append(value.reshape(16,-1))
    result=np.concatenate(parts,axis=1).mean(0)
    if not np.isfinite(result).all():raise ValueError('Nonfinite state')
    return result

class LatentReadout:
    def __init__(self,root,seed):
        root=Path(root);meta=json.loads((root/f'readout-{seed}.json').read_text());path=root/f'readout-{seed}.npz'
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if meta['seed']!=seed or meta['readout_sha256']!=self.sha256:raise ValueError('Readout identity mismatch')
        if meta['checkpoint_sha256']!=checkpoint_digest(root/f'final-{seed}'):raise ValueError('Checkpoint mismatch')
        if meta['feature_contract']!='rssm-deter-stoch-mean16-v1' or meta['threshold']!=0:raise ValueError('Unsupported readout contract')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'mean','scale','weights','bias'}:raise ValueError('Unexpected readout fields')
            self.probe={k:data[k].copy() for k in data.files}
        p=self.probe;d=len(p['mean'])
        if p['mean'].shape!=(d,) or p['scale'].shape!=(d,) or p['weights'].shape!=(d,3) or p['bias'].shape!=(3,):raise ValueError('Readout dimensions')
        if not all(np.isfinite(x).all() for x in p.values()) or not (p['scale']>0).all():raise ValueError('Invalid readout parameters')
    def predict(self,state):
        p=self.probe;x=np.asarray(state)
        if x.shape!=p['mean'].shape or not np.isfinite(x).all():raise ValueError('Invalid state features')
        scores=(x-p['mean'])/p['scale']@p['weights']+p['bias']
        if not np.isfinite(scores).all():raise ValueError('Invalid scores')
        return int(np.sum((scores>0)*np.array([1,2,4]))),scores
