"""Check actual upstream action normalization and forbid unmeasured replay actions."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from microservice.workload import WorkloadEnv,vector,next_load

def main():
    from dreamerv3 import main as upstream
    frames=[]
    for t,n in enumerate((2,8,4,10)):
        m={'checkout':dict(latency_p95=.2+t*.2,error_rate=0),'payment':dict(self_p95=.1,downstream_p95=.2),'ledger':dict(latency_p95=.15)}
        frames.append(dict(vector=[0]*10,metrics=m,requests=n,successes=n))
    row=dict(frames=frames,fault=7)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'rows.jsonl';path.write_text(json.dumps(row)+'\n')
        raw=WorkloadEnv(path);env=upstream.wrap_env(raw,None)
        obs=env.step(dict(reset=True,action=np.array([-1],np.float32)))
        np.testing.assert_array_equal(obs['vector'],vector(row,0))
        for t in range(1,5):
            action=np.array([2*next_load(row,t-1)-1],np.float32)
            obs=env.step(dict(reset=False,action=action))
            np.testing.assert_array_equal(obs['vector'],vector(row,t))
            assert obs['is_last']==(t==4)
        env.step(dict(reset=True,action=np.array([-1],np.float32)))
        try:env.step(dict(reset=False,action=np.array([-1],np.float32)))
        except ValueError as e:assert 'counterfactual' in str(e)
        else:raise AssertionError('Unmeasured transition accepted')
        changed=copy.deepcopy(row);changed['frames'][1]['metrics']['ledger']['latency_p95']=100
        np.testing.assert_array_equal(vector(row,0),vector(changed,0))
    print('WORKLOAD_PREFLIGHT_PASSED normalized workload actions and no future telemetry',flush=True)
if __name__=='__main__':main()
