import numpy as np
from microservice.latent_probe import fit_probe,predict,observations

def test_probe_recovers_independent_signals_and_does_not_refit_on_test():
    rng=np.random.default_rng(5);x=rng.normal(size=(100,5));y=np.where(x[:,:3]>0,1,-1)
    p=fit_probe(x,y);saved={k:v.copy() for k,v in p.items()}
    assert np.mean(predict(p,x)==np.sum((y>0)*[1,2,4],axis=1))>.85
    predict(p,np.full((5,5),100))
    for k in p:np.testing.assert_array_equal(p[k],saved[k])

def test_labels_do_not_change_extracted_observations():
    frames=[{'vector':[.1]*10}]*4
    assert observations({'frames':frames,'fault':0})==observations({'frames':frames,'fault':7})

def test_constant_features_do_not_make_probe_nonfinite():
    p=fit_probe(np.ones((10,4)),np.ones((10,3)))
    assert np.isfinite(p['weights']).all()
    assert np.all(predict(p,np.ones((2,4)))==7)
