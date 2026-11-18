import numpy as np
from microservice.efficiency import decisions,calibrate

def test_stopping_uses_first_confident_prefix_and_ignores_future():
    x=np.array([[[.1,.1,.1],[2,-2,2],[-4,-4,-4],[9,9,9]]])
    pred,checks=decisions(x,1.)
    assert checks.tolist()==[1] and pred.tolist()==[5]
    x[:,2:]=100
    assert decisions(x,1.)[0].tolist()==[5]

def test_fallback_and_zero_check_boundaries():
    x=np.array([[[1,-1,1],[2,2,2],[3,3,3],[-1,-1,-1]]])
    assert decisions(x,None)[1].tolist()==[3]
    assert decisions(x,None)[0].tolist()==[0]
    assert decisions(x,1.)[1].tolist()==[0]

def test_failed_development_calibration_cannot_select_cheap_rule():
    rows=[dict(fault=f) for f in range(8) for _ in range(10)]
    result=calibrate(np.ones((80,4,3)),rows)
    assert result['threshold'] is None
    assert not result['development_calibration_passed']

def test_calibration_selects_earliest_reliable_rule_not_cheap_errors():
    from microservice.mechanisms import FAULT_MASK
    rows=[dict(fault=f) for f in range(8) for _ in range(10)]
    truth=np.array([FAULT_MASK[r['fault']] for r in rows])
    signed=((truth[:,None]>>np.arange(3))&1)*2-1
    values=np.repeat(signed[:,None,:],4,axis=1).astype(float)
    values[:,0]=.1  # confident enough only at threshold 0, but all diagnoses wrong except triple.
    values[:,1]=signed*.5
    result=calibrate(values,rows)
    assert result['development_calibration_passed'] and result['threshold']==.5
    pred,checks=decisions(values,result['threshold'])
    np.testing.assert_array_equal(pred,truth)
    assert np.all(checks==1)

def test_collection_failure_retains_evidence_and_resets_controls(tmp_path):
    from unittest.mock import patch
    import pytest
    from microservice import live
    from microservice.efficiency import collect
    bad=dict(successes=0,requests=4,metrics={},vector=[0.]*10)
    path=tmp_path/'failed.jsonl'
    with patch.object(live,'wait_ready'),patch.object(live,'configure',return_value=4),patch.object(live,'batch',return_value=4),patch.object(live,'reset_measurement_window'),patch.object(live,'frame',return_value=bad),patch.object(live,'reset') as reset:
        with pytest.raises(RuntimeError,match='Invalid incident'):collect(path,8201)
        reset.assert_called_once()
    assert len(path.read_text().splitlines())==1

def test_unobserved_groups_and_later_frames_do_not_change_prefix():
    from microservice.latent_probe import observations
    import copy
    row=dict(fault=0,frames=[dict(vector=[.1]*10) for _ in range(4)])
    altered=copy.deepcopy(row)
    # After CPU check, pool and trace evidence are still unavailable.
    altered['frames'][1]['vector'][5:]=[1.9]*5
    altered['frames'][2]['vector']=[1.9]*10
    altered['frames'][3]['vector']=[1.9]*10
    assert observations(row)[:2]==observations(altered)[:2]
