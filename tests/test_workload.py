import copy
import numpy as np
from microservice.workload import vector,next_load
from microservice.load_response import response
from tests.test_regime import row

def test_future_telemetry_and_labels_cannot_affect_present():
    a=row();b=copy.deepcopy(a);b['fault']=0;b['regime']=99
    b['frames'][1]['metrics']['ledger']['latency_p95']=999
    for f in b['frames']:
        f['vector']=[999]*10
        for m in f['metrics'].values():m.update(cpu_mean=999,pool_wait_p95=999,db_mean=999)
    np.testing.assert_array_equal(vector(a,0),vector(b,0))
    b['frames'][1]['requests']=3
    assert vector(a,0)[6]!=vector(b,0)[6]  # Planned workload is intentionally available.
    assert next_load(a,3)==0

def test_response_ignores_order_and_hidden_fields():
    a=row()
    for f,n in zip(a['frames'],[2,8,2,8]):f['requests']=n
    b=copy.deepcopy(a);b['frames'].reverse();b['fault']=0
    for f in b['frames']:
        for m in f['metrics'].values():m.update(cpu_mean=999,pool_wait_p95=999)
    np.testing.assert_array_equal(response(a),response(b))
