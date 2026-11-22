import copy
import numpy as np
import pytest
from tests.fixtures import incident
from microservice.probe_target import frames
from microservice.triage_audit import features,baseline_fit,control_predictions


def test_triage_residuals_only_use_predictions_whose_outcomes_have_arrived():
    a=incident();b=copy.deepcopy(a);b['fault']=7
    pred=np.array([[1.,2.,3.,4.]]);changed=pred.copy();changed[:,2:]=999
    for f in frames(b)[6:]:
        for m in f['metrics'].values():m.update(latency_p95=900,self_p95=800,downstream_p95=700)
    np.testing.assert_array_equal(features([a],6,pred),features([b],6,changed))
    assert features([a],6,pred).shape==(1,6*14+4)
    assert features([a],4,pred).shape==(1,4*14)
    with pytest.raises(ValueError):features([a],9,pred)


def test_prefix_control_selection_is_training_only_and_repeatable():
    rows=[incident(repeat=i) for i in range(8)]
    model,selected,oof=baseline_fit(rows)
    assert set(selected)=={'4','5','6','7'}
    pred,target=control_predictions(rows,model,selected)
    assert pred.shape==target.shape==(8,4)
    np.testing.assert_allclose(pred,target,atol=1e-6)
    assert set(oof)=={'persistence','matching_probe','regression'}
