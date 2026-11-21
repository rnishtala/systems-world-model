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


def test_measurement_failure_is_reported_without_retry_or_partial_row(tmp_path,capsys):
    from microservice.collect import collect
    path=tmp_path/'rows.jsonl';case=dict(fault=3,repeat=0,regime=1)
    with patch('microservice.collect.training_cases',return_value=[case]), patch('microservice.collect.configure_training'), patch('microservice.collect.live.wait_ready'), patch('microservice.collect.live.reset') as reset, patch('microservice.collect.batch',return_value=9), patch('microservice.collect.frame',side_effect=RuntimeError('count mismatch')) as frame:
        with pytest.raises(RuntimeError,match='count mismatch'):collect(path,True,123,'ledger-first')
    assert frame.call_count==1 and path.read_text()==''
    assert reset.call_count==2
    line=capsys.readouterr().out.split('PROBE_RANGE_FAILURE ',1)[1]
    details=json.loads(line)
    assert details['seed']==123 and details['episode']==0 and details['fault']==3
    assert details['schedule']=='ledger-first'


def test_measurement_error_identifies_service_counts_and_successes():
    from microservice.ledger_probe import frame
    metrics=dict(count=2,latency_p95=.1,self_p95=.1,downstream_p95=0.,error_rate=0.)
    with patch('microservice.ledger_probe.live.reset_measurement_window'), patch('microservice.ledger_probe.batch',return_value=3), patch('microservice.ledger_probe.live.request',return_value=metrics):
        with pytest.raises(RuntimeError) as error:frame('checkout',3)
    details=json.loads(str(error.value).split(': ',1)[1])
    assert details['service']=='checkout'
    assert details['target']=='checkout'
    assert details['expected_count']==3 and details['actual_count']==2 and details['successes']==3
