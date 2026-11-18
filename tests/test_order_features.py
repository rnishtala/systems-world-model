from types import SimpleNamespace
import numpy as np
from microservice.order_features import observed,force_action,ORDERS

def test_order_reveals_only_requested_group_and_never_future_frames():
    row=dict(fault=7,frames=[dict(vector=[float(t+1)/100]*10) for t in range(4)])
    trace=observed(row,(3,))[-1]
    assert trace[10:13].tolist()==[0,0,1]
    assert np.all(trace[3:7]==0) and np.all(trace[7:10]>0)
    before=observed(row,(3,)).copy();row['frames'][2]['vector']=[1.]*10
    np.testing.assert_array_equal(observed(row,(3,)),before)
    row['fault']=0;np.testing.assert_array_equal(observed(row,(3,)),before)

def test_action_override_is_actual_check_and_does_not_mutate_parent():
    old=SimpleNamespace(sharding='fake');carry=('state',{'action':[old],'other':42})
    calls=[]
    def put(value,sharding):calls.append((int(value),sharding));return value
    new=force_action(carry,3,put)
    assert calls==[(3,'fake')] and int(new[-1]['action'][0])==3
    assert carry[-1]['action'][0] is old and new[-1]['other']==42

def test_shared_order_prefixes_have_identical_observations():
    row=dict(frames=[dict(vector=[.1]*10) for _ in range(4)])
    for a in ORDERS:
        for b in ORDERS:
            for t in range(4):
                if a[:t]==b[:t]:np.testing.assert_array_equal(observed(row,a)[:t+1],observed(row,b)[:t+1])
