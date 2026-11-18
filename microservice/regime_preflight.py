"""Exercise the actual upstream environment before live collection."""
import json,tempfile
from pathlib import Path
import numpy as np
from microservice.regime import LogEnv,vector
from microservice.mechanisms import FAULT_MASK

def main():
    frames=[]
    for t in range(4):
        metrics={'checkout':dict(latency_p95=.4+t*.4,error_rate=0),
                 'payment':dict(self_p95=.2,downstream_p95=.5), 'ledger':dict(latency_p95=.3)}
        frames.append(dict(vector=[0]*10,metrics=metrics,requests=4,successes=4))
    rows=[dict(frames=frames,fault=f) for f in range(8)]
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'rows.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
        env=LogEnv(p,71,True)
        for _ in range(20):
            for t,action in enumerate([dict(reset=True,action=0)]+[dict(reset=False,action=0)]*3):
                obs=env.step(action)
                np.testing.assert_array_equal(obs['vector'],vector(rows[0],t))
                mask=FAULT_MASK[env.sim.fault]
                np.testing.assert_array_equal(obs['mechanism_target'],[(mask>>i)&1 for i in range(3)])
        assert obs['vector'][0]>np.log1p(.6/.1)
    print('REGIME_PREFLIGHT_PASSED 80 upstream environment observations',flush=True)
if __name__=='__main__':main()
