import numpy as np
from microservice.regularized_head import fit_head,predict,folds

def test_transform_is_fitted_on_training_only():
 x=np.random.default_rng(3).normal(size=(80,10));y=np.where(x[:,:3]>0,1.,-1.)
 h=fit_head(x,y,8,.01);saved={k:v.copy() for k,v in h.items()}
 assert np.isfinite(predict(h,np.full((4,10),1000))).all()
 for k in h:np.testing.assert_array_equal(h[k],saved[k])
 np.testing.assert_allclose(h['mean'],x.mean(0))

def test_folds_are_disjoint_and_withhold_whole_pair_families():
 target=np.repeat([0,1,2,4,3,5,6],20);fs=folds(target)
 assert np.all(sum(test.astype(int) for _,test,k in fs if k=='stratified')==1)
 for train,test,kind in fs:
  assert not np.any(train&test) and np.all(train|test)
  if kind=='held_pair':assert len(set(target[test]))==1 and set(target[test]).isdisjoint(target[train])

def test_constant_features_and_rank_bound_are_safe():
 h=fit_head(np.ones((8,12)),np.ones((8,3)),64,1.)
 assert h['components'].shape[1]==7
 assert np.all(predict(h,np.ones((2,12)))==7)

def test_stronger_regularization_shrinks_weights():
 x=np.random.default_rng(8).normal(size=(50,9));y=np.where(x[:,:3]>0,1.,-1.)
 assert np.linalg.norm(fit_head(x,y,0,100)['weights'])<np.linalg.norm(fit_head(x,y,0,.01)['weights'])

def test_hybrid_prefers_raw_when_latent_has_no_information():
 from microservice.regularized_head import select_hybrid
 target=np.repeat([0,1,2,4,3,5,6],10);y=(((target[:,None]>>np.arange(3))&1)*2-1).astype(float)
 raw=np.concatenate([y,np.zeros((len(y),5))],axis=1)
 h,selection=select_hybrid(np.ones((len(y),20)),raw,y)
 assert selection['chosen']['rank']==0 and selection['chosen']['latent_weight']==0
 assert selection['chosen']['score']==1
 assert np.all(predict(h,np.concatenate([np.ones((len(y),20)),raw],axis=1))==target)
