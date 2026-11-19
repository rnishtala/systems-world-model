import numpy as np
import microservice.probe_retrain as experiment
from microservice.latent_probe import labels
from microservice.partial_readout import diagnose


def test_gate_uses_refitted_old_model_and_its_own_features(monkeypatch):
    rows=[dict(fault=f,regime=g) for g in (0,1) for f in range(8) for _ in range(4)]
    truth=diagnose(labels(rows));new=truth.copy();old=truth.copy();refit=truth.copy()
    no_pool=np.flatnonzero((truth&2)==0)
    old[no_pool[:3]]|=2;refit[no_pool[:2]]|=2
    def readout(model,features):
        if model=='original':
            assert features is old_features
            p=old
        elif model=='refit':
            assert features is old_features
            p=refit
        else:p=new
        return np.where((p[:,None] & np.array([1,2,4]))>0,1.,-1.)
    old_features=np.zeros((64,640));xs={f'{n}_{t}':np.ones((64,1)) for n in ('latent','history') for t in (5,7)}
    monkeypatch.setattr(experiment,'score',readout)
    models={n:n for n in ('latent6','latent8','history6','history8')}
    result,_=experiment.evaluate(xs,rows,models,'original','refit',old_features)
    assert result['feasibility_gate'] and result['world_model_retrained']
    assert result['methods']['refit_latent6']['correct']==62
    # If the refitted comparator is equally good, beating only the original is insufficient.
    refit[:]=truth
    result,_=experiment.evaluate(xs,rows,models,'original','refit',old_features)
    assert not result['improvement_gate'] and not result['feasibility_gate']
