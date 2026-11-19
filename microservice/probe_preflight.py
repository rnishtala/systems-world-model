"""Exercise actual upstream action normalization before any training."""
import copy,json
from pathlib import Path
import numpy as np
from microservice.probe_target import ProbeEnv,vector,next_action

def main():
    from dreamerv3 import main as upstream
    path=Path('collection/train.jsonl');raw=ProbeEnv(path);env=upstream.wrap_env(raw,None)
    env.step(dict(reset=True,action=np.full(2,-1,np.float32)));row=copy.deepcopy(raw.row)
    for t in range(1,9):
        obs=env.step(dict(reset=False,action=2*next_action(row,t-1)-1))
        np.testing.assert_array_equal(obs['vector'],vector(row,t));assert obs['is_last']==(t==8)
    env.step(dict(reset=True,action=np.full(2,-1,np.float32)))
    wrong=next_action(raw.row,0).copy();wrong[0]=1-wrong[0]
    try:env.step(dict(reset=False,action=2*wrong-1))
    except ValueError as e:assert 'counterfactual' in str(e)
    else:raise AssertionError('Unmeasured action accepted')
    changed=copy.deepcopy(row);changed['fault']=99;changed['base'][1]['metrics']['ledger']['latency_p95']=999
    np.testing.assert_array_equal(vector(row,0),vector(changed,0))
    print('PROBE_PREFLIGHT_PASSED actual wrapper, two action dimensions, label/future isolation',flush=True)
if __name__=='__main__':main()
