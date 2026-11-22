import copy
import numpy as np
import pytest
from tests.fixtures import incident
from microservice.forecast import CUTS,history,design,controls,fit,predict,summarize
from microservice.probe_target import frames

def test_forecast_inputs_exclude_future_telemetry_and_fault_labels():
    for cut in CUTS:
        a=incident();b=copy.deepcopy(a);b['fault']=999;b['seed']=0
        for frame in frames(b)[cut:]:
            for metrics in frame['metrics'].values():
                metrics['latency_p95']=900;metrics['self_p95']=800;metrics['downstream_p95']=700
        np.testing.assert_array_equal(history(a,cut),history(b,cut))
        np.testing.assert_array_equal(design(history(a,cut)),design(history(b,cut)))
        assert controls(history(a,cut))==controls(history(b,cut))


def test_forecast_inputs_include_known_next_action():
    a=incident();b=copy.deepcopy(a);cut=4
    frame=frames(b)[cut];frame['requests']=frame['successes']=11
    assert not np.array_equal(history(a,cut)[-1,6:8],history(b,cut)[-1,6:8])


def test_matching_control_uses_past_target_and_load_only():
    h=np.zeros((4,14));h[:,0]=[1,2,3,4]
    h[:,4:6]=[[0,.25],[0,.75],[1,.25],[0,.25]];h[-1,6:8]=[1,.25]
    assert controls(h)==dict(persistence=4,matching_probe=3)
    h[-1,6:8]=[1,.99];assert controls(h)['matching_probe']==4


def test_regression_predicts_heldout_linear_relationship():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,3));y=2*x[:,0]-x[:,1]+.3
    m=fit(x[:80],y[:80],.01)
    np.testing.assert_allclose(predict(m,x[80:]),y[80:],atol=.003)


def test_scoring_uses_incidents_and_preselected_control():
    y=np.ones(8);pred=dict(dreamer=np.ones(8)*1.1,persistence=np.ones(8)*1.5,matching_probe=np.ones(8)*1.2,regression=np.ones(8)*1.05)
    s=summarize(y,pred,'persistence',2)
    assert s['incidents']==2 and s['predictions']==8 and s['selected_control']=='persistence'
    assert s['improvement_gate'] and s['paired_incident_bootstrap_95'][1]<0
    assert not summarize(y,pred,'regression',2)['improvement_gate']
    pred['dreamer'][0]=np.nan
    with pytest.raises(ValueError):summarize(y,pred,'persistence',2)
