from unittest.mock import patch
from microservice.collect import training_cases,configure_training

def test_new_coverage_has_balanced_independent_ranges():
    cases=training_cases(9501);assert cases==training_cases(9501) and len(cases)==64
    for fault in range(8):
        rs=[r for r in cases if r['fault']==fault]
        assert len(rs)==8 and sum(r['regime']==0 for r in rs)==4
        assert sum(r['cpu']>.05 for r in rs)==4
        assert sum(r['delay']>.23 for r in rs)==4
        assert sum(r['hold']>.035 for r in rs)==4
        assert sum(r['pool']==1 for r in rs)==4

def test_injected_mechanisms_match_labels_and_hold_is_not_a_label():
    for c in training_cases(9501):
        with patch('microservice.collect.live.reset') as reset,patch('microservice.collect.live.request') as request:
            configure_training(c);reset.assert_called_once()
            ledger=request.call_args_list[0].args[2]
            assert ledger['db_hold']==c['hold']
            assert ('pool_size' in ledger)==(c['fault'] in (2,5,6,7))
            assert ('delay_seconds' in ledger)==(c['fault'] in (3,4,6,7))
            assert (request.call_count==2)==(c['fault'] in (1,4,5,7))


import json
import pytest

@pytest.mark.parametrize('schedule,repeat,order',[
    ('mixed',0,['checkout','ledger']),('mixed',2,['ledger','checkout']),
    ('ledger-first',0,['ledger','checkout']),('ledger-first',2,['ledger','checkout'])])
def test_collector_measures_requested_order(tmp_path,schedule,repeat,order):
    from microservice.collect import collect
    case=dict(fault=0,repeat=repeat,regime=0)
    path=tmp_path/'rows.jsonl'
    with patch('microservice.collect.training_cases',return_value=[case]), patch('microservice.collect.configure_training'), patch('microservice.collect.live.wait_ready'), patch('microservice.collect.live.reset') as reset, patch('microservice.collect.batch',return_value=9), patch('microservice.collect.frame',side_effect=lambda target,n:dict(target=target,requests=n)) as frame:
        collect(path,True,123,schedule)
    assert [c.args[0] for c in frame.call_args_list]==['checkout']*4+[t for t in order for _ in range(2)]
    row=json.loads(path.read_text());assert row['order']==order
    assert [f['target'] for t in row['order'] for f in row['extra'][t]]==[t for t in order for _ in range(2)]
    assert reset.call_count==2


def test_invalid_schedule_cannot_touch_lab(tmp_path):
    from microservice.collect import collect
    with patch('microservice.collect.live.wait_ready') as ready:
        with pytest.raises(ValueError):collect(tmp_path/'rows',True,123,'arbitrary')
        ready.assert_not_called()
