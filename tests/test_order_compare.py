import numpy as np
from microservice.order_compare import choose_order,KEYS

def run_with_correct_prefix(prefix,calibrated=True):
    families=np.repeat(np.arange(8),8);y=((families[:,None]>>np.arange(3))&1)*2-1
    scores=np.repeat(y[:,None,:],4,axis=1)*.1;scores[:,prefix:]=y[:,None,:]*2
    return dict(inner_scores=scores,policy=dict(threshold=1.,calibration_passed=calibrated))

def test_order_selection_uses_inner_efficiency_and_lexicographic_ties():
    families=np.repeat(np.arange(8),8);runs={k:run_with_correct_prefix(3) for k in KEYS}
    runs['312']=run_with_correct_prefix(1);runs['321']=run_with_correct_prefix(1)
    key,threshold,_=choose_order(runs,families,families)
    assert key=='312' and threshold==1.

def test_failed_calibration_forces_full_check_fallback():
    families=np.repeat(np.arange(8),8);runs={k:run_with_correct_prefix(1,False) for k in KEYS}
    key,threshold,_=choose_order(runs,families,families)
    assert key=='123' and threshold is None

def test_order_choice_cannot_see_outer_predictions():
    families=np.repeat(np.arange(8),8);runs={k:run_with_correct_prefix(3) for k in KEYS}
    runs['231']=run_with_correct_prefix(2)
    before=choose_order(runs,families,families)
    for key,run in runs.items():run['scores']=np.full((16,4,3),1000 if key=='123' else -1000)
    after=choose_order(runs,families,families)
    assert before==after
