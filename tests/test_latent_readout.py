import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from microservice.latent_readout import LatentReadout,checkpoint_digest,state_features

class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'final-23').mkdir();(self.root/'final-23/weights').write_bytes(b'checkpoint')
        self.probe=dict(mean=np.zeros(3),scale=np.ones(3),weights=np.eye(3),bias=np.zeros(3));self.save()
    def save(self):
        np.savez_compressed(self.root/'readout-23.npz',**self.probe)
        self.meta=dict(seed=23,readout_sha256=hashlib.sha256((self.root/'readout-23.npz').read_bytes()).hexdigest(),checkpoint_sha256=checkpoint_digest(self.root/'final-23'),feature_contract='rssm-deter-stoch-mean16-v1',threshold=0.)
        (self.root/'readout-23.json').write_text(json.dumps(self.meta))
    def test_predictions_use_fixed_coefficients(self):
        r=LatentReadout(self.root,23);saved={k:v.copy() for k,v in r.probe.items()}
        self.assertEqual(r.predict([1,-1,2])[0],5);self.assertEqual(r.predict([0,0,0])[0],0)
        for k,v in saved.items():np.testing.assert_array_equal(r.probe[k],v)
    def test_corrupt_artifact_rejected(self):
        with (self.root/'readout-23.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'identity'):LatentReadout(self.root,23)
    def test_different_checkpoint_rejected(self):
        (self.root/'final-23/weights').write_bytes(b'other model')
        with self.assertRaisesRegex(ValueError,'Checkpoint'):LatentReadout(self.root,23)
    def test_invalid_parameters_and_state_rejected(self):
        r=LatentReadout(self.root,23)
        for x in ([1,2],[1,np.nan,3]):
            with self.assertRaises(ValueError):r.predict(x)
        self.probe['scale'][0]=0;self.save()
        with self.assertRaisesRegex(ValueError,'parameters'):LatentReadout(self.root,23)
    def test_feature_layout_matches_confirmed_extractor(self):
        dyn={'deter':np.arange(16*4).reshape(16,4),'stoch':np.arange(16*2*3).reshape(16,2,3)}
        expected=np.concatenate([np.asarray(dyn[k]).reshape(16,-1) for k in ('deter','stoch')],axis=1).mean(0)
        np.testing.assert_array_equal(state_features(dyn),expected)
        with self.assertRaises(ValueError):state_features({'deter':np.ones((8,4)),'stoch':np.ones((16,2,3))})
    def test_wrong_feature_contract_rejected(self):
        self.meta['feature_contract']='other';(self.root/'readout-23.json').write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError,'contract'):LatentReadout(self.root,23)
    def test_serving_package_excludes_examples_and_replay(self):
        from microservice.package_latent import main
        (self.root/'train-23/replay').mkdir(parents=True);(self.root/'train-23/replay/example').write_text('training data')
        (self.root/'train-23/config.yaml').write_text('config');(self.root/'training-23.json').write_text('{}')
        ev=self.root/'eval';ev.mkdir();(ev/'test.jsonl').write_text('held-out data')
        (ev/'fitted.json').write_text(json.dumps(dict(checkpoint_sha256=checkpoint_digest(self.root/'final-23'),calibration_sha256='calibration')))
        np.savez_compressed(ev/'probes.npz',**{'latent_'+k:v for k,v in self.probe.items()})
        out=self.root/'package/serving'
        with patch('sys.argv',['package','--models',str(self.root),'--evaluation',str(ev),'--out',str(out)]):main()
        self.assertFalse((out/'train-23/replay').exists());self.assertFalse((out/'test.jsonl').exists());self.assertEqual(LatentReadout(out,23).predict([1,-1,2])[0],5)
if __name__=='__main__':unittest.main()
