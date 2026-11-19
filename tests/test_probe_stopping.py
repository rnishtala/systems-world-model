import numpy as np
from scripts.evaluate_probe_stopping import decide,outer

def test_stop_uses_first_eligible_prefix_and_ignores_future():
    x=np.zeros((2,8,3));x[0,2]=[.6,-.7,.8];x[0,3:]=[-10,10,-10];x[1,-1]=[-.1,.1,-.1]
    pred,stop=decide(x,.5)
    np.testing.assert_array_equal(stop,[2,7]);np.testing.assert_array_equal(pred,[5,2])
    x[0,3:]=100
    np.testing.assert_array_equal(decide(x,.5)[0],pred)
    np.testing.assert_array_equal(decide(x,None)[1],[7,7])

def test_outer_predictions_and_policy_do_not_use_held_labels():
    rng=np.random.default_rng(13);families=np.tile(np.arange(8),4);groups=np.repeat(np.arange(4),8)
    y=((families[:,None]>>np.arange(3))&1)*2-1
    xs=[rng.normal(size=(32,3)) for _ in range(8)];cost=np.tile(np.arange(1,9),(32,1))
    train=np.where(groups!=3)[0];held=np.where(groups==3)[0]
    a,aa=outer(xs,y,groups,families,cost,train,held)
    changed=y.copy();changed[held]*=-1
    b,bb=outer(xs,changed,groups,families,cost,train,held)
    np.testing.assert_array_equal(a,b);assert aa==bb
