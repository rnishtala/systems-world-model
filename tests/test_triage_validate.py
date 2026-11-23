import numpy as np
import pytest
from tests.fixtures import incident
from microservice.triage_validate import freeze,load,paired


def test_frozen_readouts_roundtrip_and_detect_tampering(tmp_path):
    rows=[incident(fault=i,repeat=i%4) for i in range(8)]
    forecasts=np.zeros((8,4));out=tmp_path/'readouts'
    frozen=freeze(rows,forecasts,forecasts,out,{'checkpoint_sha256':'fixed'})
    loaded,models=load(out)
    assert loaded==frozen and len(models)==15
    assert frozen['checkpoint_sha256']=='fixed'
    with (out/'readouts.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):load(out)


def test_paired_accuracy_counts_disagreements_only():
    assert paired([0,1,2],[0,0,0],np.array([0,1,2]))==dict(candidate_only_correct=2,control_only_correct=0,exact_p=.5)
    assert paired([0],[0],np.array([0]))['exact_p']==1
