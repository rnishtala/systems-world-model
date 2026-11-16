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
class SupervisionTests(unittest.TestCase):
    def test_targets_removed_without_mutating_input(self):
        from benchmark.supervision import telemetry,TARGET
        obs={'vector':np.array([1.,2.]),TARGET:np.array([0.,1.])}
        cleaned=telemetry(obs)
        self.assertEqual(set(cleaned),{'vector'})
        self.assertIn(TARGET,obs)
        self.assertIs(cleaned['vector'],obs['vector'])

class ReadoutTests(unittest.TestCase):
    def test_rejects_test_inputs_before_loading_model(self):
        import json,sys
        from unittest.mock import patch
        from benchmark.readout import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'manifest.json').write_text(json.dumps([{'id':'held-out','split':'test'}]))
            argv=['readout','--root',str(root/'nonexistent-model'),'--data',str(root),'--answers',str(root/'no-labels'),'--out',str(root/'out')]
            with patch.object(sys,'argv',argv),self.assertRaises(AssertionError):main()
    def test_validation_training_rejects_test_manifest(self):
        import json,sys
        from unittest.mock import patch
        from benchmark.dreamer import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'manifest.json').write_text(json.dumps([{'id':'held-out','split':'test'}]))
            argv=['dreamer','--data',str(root),'--train-answers',str(root/'no-labels'),'--out',str(root/'out'),'--validation-only']
            with patch.object(sys,'argv',argv),self.assertRaises(AssertionError):main()
    def test_confirmation_rejects_test_labels_before_loading_model(self):
        import json,sys
        from unittest.mock import patch
        from benchmark.confirmation import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[{'id':'a','split':'train'},{'id':'b','split':'test'}]
            (root/'manifest.json').write_text(json.dumps(rows));(root/'answers.json').write_text(json.dumps(rows))
            argv=['confirm','--root',str(root/'no-model'),'--data',str(root),'--answers',str(root/'answers.json'),'--out',str(root/'out')]
            with patch.object(sys,'argv',argv),self.assertRaises(AssertionError):main()
    def test_confirmation_scoring_without_validation(self):
        truth=[dict(id='a',service='a',fault='cpu',split='test')]
        result=score([dict(id='a',ranks=['a'])],truth)
        self.assertEqual(result['test']['ac_at_k']['1'],1.)
        self.assertNotIn('validation',result)
    def test_ridge_predicts_without_using_validation_labels(self):
        from benchmark.readout import ridge
        x=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        y=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        mean,scale,w,bias=ridge(x,y,1.)
        scores=(np.array([[0.2,0.2],[2.8,2.8]])-mean)/scale@w+bias
        self.assertEqual(scores.argmax(1).tolist(),[0,1])
if __name__=="__main__":unittest.main()
