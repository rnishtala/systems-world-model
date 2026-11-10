import json
import numpy as np
from microservice.supervision import SupervisedEnv,TARGET,telemetry
from microservice.environment import EmpiricalEnv

def test_labels_do_not_change_telemetry(tmp_path):
    rows=[dict(fault=f,frames=[dict(vector=[.1]*10,successes=1,requests=1)]*4) for f in range(8)]
    p=tmp_path/'data.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    supervised=SupervisedEnv(p,17);base=EmpiricalEnv(p,17,True)
    for _ in range(20):
        for act in [dict(reset=True,action=0)]+[dict(reset=False,action=i) for i in (1,2,3)]:
            a=supervised.step(act);b=base.step(act)
            assert set(telemetry(a))==set(b)
            for k in b:np.testing.assert_array_equal(a[k],b[k])
            assert a[TARGET].shape==(3,)

def test_target_removed_without_mutating_input():
    obs={'vector':np.zeros(14),TARGET:np.ones(3)}
    assert TARGET not in telemetry(obs) and TARGET in obs
