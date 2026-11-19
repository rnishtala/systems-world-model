import copy,json
import numpy as np
import pytest
from microservice.validate import check_test,summarize,load_readouts
from microservice.readout import labels,diagnose,fit,score,choose
from tests.fixtures import incident

def cases():return [incident(seed=20,fault=f,repeat=r,regime=g) for g in (0,1) for f in range(8) for r in range(4)]

def test_rejects_leakage_and_incomplete_confirmation():
    rows=cases();frozen=dict(training_seeds=[10],utc=0)
    check_test(rows,frozen)
    with pytest.raises(ValueError,match='overlap'):check_test(rows,dict(frozen,training_seeds=[20]))
    with pytest.raises(ValueError,match='after'):check_test(rows,dict(frozen,utc=2))
    with pytest.raises(ValueError,match='balanced'):check_test(rows[:-1],frozen)

def test_pairing_and_pool_errors():
    rows=cases();truth=diagnose(labels(rows));pred={n:truth.copy() for n in ('latent6','latent8','history6','history8','baseline6')}
    pred['latent6'][12]|=2;pred['baseline6'][12:15]|=2
    summary=summarize(pred,rows)
    assert summary['methods']['latent6']['correct']==63
    assert summary['false_pool']['latent6']==1
    assert summary['bit_errors']['latent6']==[0,1,0]
    assert summary['paired']['baseline6']['new_only_correct']==2

def test_readout_hash_guard(tmp_path):
    (tmp_path/'readouts.npz').write_bytes(b'changed')
    (tmp_path/'frozen.json').write_text(json.dumps(dict(readouts_sha256='wrong')))
    with pytest.raises(ValueError,match='changed'):load_readouts(tmp_path)

def test_training_selection_deterministic():
    y=labels(cases());x=y.copy();groups=np.tile(np.arange(4),16)
    first=choose(x,y,groups);assert first==choose(x,y,groups)
    assert np.array_equal(diagnose(score(fit(x,y,first[0]),x)),diagnose(y))
