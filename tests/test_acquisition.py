import unittest
import numpy as np
from benchmark.acquisition import ObservationOracle,groups,retrieval_action,fit_shared,rank,random_order

class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.cols=['a_cpu','b_cpu','c_cpu','d_cpu','e_cpu'];self.g=groups(self.cols,['a','b','c','d','e'])
    def test_hidden_observations_cannot_change_visible_state(self):
        first=ObservationOracle(np.arange(10,dtype=float),self.g)
        changed=np.arange(10,dtype=float);changed[[1,2,3,4,6,7,8,9]]=999
        other=ObservationOracle(changed,self.g)
        np.testing.assert_array_equal(first.observe(),other.observe())
        np.testing.assert_array_equal(first.reveal(0),other.reveal(0))
        self.assertTrue((first.observe()[[1,2,3,4,6,7,8,9]]==0).all())
        with self.assertRaises(ValueError):first.reveal(0)
        with self.assertRaises(ValueError):first.reveal(5)
    def test_retrieval_reads_only_observed_test_coordinates(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(10,10));y=np.arange(10)%5
        seen=np.array([True,False,False,False,False]);obs=np.zeros(15);changed=obs.copy();changed[[1,2,3,4,6,7,8,9]]=999
        a=retrieval_action(obs,seen,x,y,self.g);b=retrieval_action(changed,seen,x,y,self.g)
        self.assertEqual(a,b);self.assertNotEqual(a,0)
    def test_shared_classifier_and_distinct_queries(self):
        x=np.concatenate([np.eye(5)*5,np.ones((5,5))],1);y=np.arange(5)
        probe=fit_shared(x,y,self.g)
        for i in range(5):
            oracle=ObservationOracle(x[i],self.g)
            for action in range(5):oracle.reveal(action)
            self.assertEqual(rank(probe,oracle.observe())[0],i)
        self.assertEqual(sorted(random_order('case',0)),list(range(5)))
        self.assertEqual(random_order('case',0),random_order('case',0))
if __name__=='__main__':unittest.main()
