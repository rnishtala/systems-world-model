import numpy as np
from microservice.partial_readout import fit,score,kernel,folds,outer_fit,diagnose
from microservice.latent_probe import fit_probe
from microservice.efficiency import scores

def test_linear_model_matches_existing_ridge_and_does_not_refit():
    rng=np.random.default_rng(22);x=rng.normal(size=(12,8));x[:,0]=1;y=rng.choice([-1,1],size=(12,3))
    model=fit(x,y,dict(kind='linear',penalty=1.));mean=model['mean'].copy()
    z=rng.normal(size=(5,8));np.testing.assert_allclose(score(model,z),scores(fit_probe(x,y),z),atol=1e-10)
    score(model,z*1e6);np.testing.assert_array_equal(model['mean'],mean)

def test_rbf_can_recover_nonlinear_xor_and_kernel_is_symmetric():
    x=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float);y=np.repeat((x[:,0]*x[:,1])[:,None],3,axis=1)
    m=fit(x,y,dict(kind='rbf',gamma=1.,penalty=.01))
    np.testing.assert_array_equal(diagnose(score(m,x)),diagnose(y))
    np.testing.assert_allclose(kernel(x,x,1.),kernel(x,x,1.).T)

def test_outer_predictions_and_threshold_ignore_outer_labels():
    rng=np.random.default_rng(44);families=np.repeat(np.arange(8),5);y=((families[:,None]>>np.arange(3))&1)*2-1
    features=[rng.normal(size=(40,3)) for _ in range(4)]
    test=folds(families,5,1)[0];train=np.setdiff1d(np.arange(40),test);inner=folds(families[train],4,2)
    a=outer_fit(features,y,families,train,test,inner);changed=y.copy();changed[test]*=-1
    b=outer_fit(features,changed,families,train,test,inner)
    np.testing.assert_array_equal(a['scores'],b['scores']);assert a['policy']==b['policy'] and a['selected']==b['selected']

def test_nested_splits_partition_all_cases_without_overlap():
    families=np.repeat(np.arange(8),10);outer=folds(families,5,8301)
    assert sorted(np.concatenate(outer))==list(range(80))
    for k,test in enumerate(outer):
        train=np.setdiff1d(np.arange(80),test)
        for held in folds(families[train],4,8302+k):assert not set(train[held])&set(test)
