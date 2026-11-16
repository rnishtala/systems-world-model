import tempfile,unittest
from pathlib import Path
import numpy as np
from benchmark.evaluate import rank_case,score
from benchmark.dreamer import sequence
class BenchmarkTests(unittest.TestCase):
    def test_time_is_not_ranked_and_missing_finite(self):
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/"incident.npz";t=np.arange(-300,300)
            good=np.tile([1.,2.,3.],200);bad=good+(t>=0)*100
            np.savez(path,time=t,values=np.stack([good,bad,np.full(600,np.nan)],1),columns=np.array(["healthy_cpu","bad_cpu","missing_cpu"]))
            ranks,_=rank_case(path);self.assertEqual(ranks[0],"bad")
            x=sequence(path,["bad_cpu","healthy_cpu","absent_cpu"])
            self.assertEqual(x.shape,(20,6));self.assertTrue(np.isfinite(x).all());self.assertTrue((x[:,5]==0).all())
    def test_scoring_uses_all_expected_cases(self):
        truth=[dict(id=str(i),service="a",fault="cpu",split=s) for i,s in enumerate(["train","validation","test"])]
        pred=[dict(id=str(i),ranks=["b","a"]) for i in range(3)]
        self.assertEqual(score(pred,truth)["test"]["ac_at_k"]["2"],1)
        with self.assertRaises(ValueError):score(pred[:2],truth)
if __name__=="__main__":unittest.main()
