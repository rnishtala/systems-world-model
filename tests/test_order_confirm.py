import numpy as np
from microservice.order_confirm import load_readout
from microservice.partial_readout import fit,score

def test_serialized_frozen_readout_matches_in_memory_without_mutation(tmp_path):
    rng=np.random.default_rng(8401);x=rng.normal(size=(20,7));y=rng.choice([-1,1],size=(20,3));config=dict(kind='rbf',gamma=1.,penalty=.01)
    original=fit(x,y,config)
    path=tmp_path/'readouts.npz';np.savez_compressed(path,**{f'latent_0_{k}':original[k] for k in ['mean','scale','bias','z','dual']})
    before=path.read_bytes();params=np.load(path);policy=dict(policies={'latent':dict(readouts=[config])})
    restored=load_readout(params,policy,'latent',0);new=rng.normal(size=(5,7))
    np.testing.assert_array_equal(score(original,new),score(restored,new))
    assert path.read_bytes()==before
