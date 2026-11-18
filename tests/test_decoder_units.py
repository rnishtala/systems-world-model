import numpy as np
from microservice.decoder_units import observation_prediction

def test_roundtrip_including_negative_values():
    x=np.array([-5.,-1.,0.,.5,2.5])
    encoded=np.sign(x)*np.log1p(abs(x))
    np.testing.assert_allclose(observation_prediction(encoded,True,np),x,atol=1e-12)
    np.testing.assert_array_equal(observation_prediction(x,False,np),x)

def test_inverse_precedes_particle_average():
    particles=np.log1p(np.array([[1.,3.],[5.,7.]]))
    expected=np.array([3.,5.])
    np.testing.assert_allclose(observation_prediction(particles,True,np).mean(0),expected)
    assert not np.allclose(observation_prediction(particles.mean(0),True,np),expected)
