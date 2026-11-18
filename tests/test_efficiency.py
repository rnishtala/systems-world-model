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
