import unittest
import numpy as np
from benchmark.alignment import imagined_inputs

class AlignmentTests(unittest.TestCase):
    def test_only_queried_service_is_changed(self):
        obs=np.zeros(10);obs[[0,2,4,6,8]]=[7,9,1,1,1]
        particles=np.full((32,10),999.)
        result=imagined_inputs(obs,particles,np.array([1,3,5,7]),1,8)
        for j in [0,2,4,6,8]:np.testing.assert_array_equal(result[:,j],np.repeat(obs[j],32))
        self.assertTrue((result[:,[1,3,5,7]]==999).all());self.assertTrue((result[:,9]==1).all())
        self.assertEqual(obs[1],0)
    def test_clipping_does_not_use_nonquery_forecasts(self):
        obs=np.zeros(10);p=np.full((32,10),999.)
        low=np.zeros(8);high=np.array([2,2,2,2,1,1,1,1])
        a=imagined_inputs(obs,p,np.array([1,3,5,7]),1,8,low,high)
        p[:,[0,2,4,6,8,9]]=-999
        b=imagined_inputs(obs,p,np.array([1,3,5,7]),1,8,low,high)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a[:,[1,3]]==2).all());self.assertTrue((a[:,[5,7,9]]==1).all())
        self.assertTrue((a[:,[0,2,4,6,8]]==0).all())
