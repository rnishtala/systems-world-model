from unittest.mock import patch
from microservice.probe_range import training_cases,configure_training

def test_new_coverage_has_balanced_independent_ranges():
    cases=training_cases();assert cases==training_cases() and len(cases)==64
    for fault in range(8):
        rs=[r for r in cases if r['fault']==fault]
        assert len(rs)==8 and sum(r['regime']==0 for r in rs)==4
        assert sum(r['cpu']>.05 for r in rs)==4
        assert sum(r['delay']>.23 for r in rs)==4
        assert sum(r['hold']>.035 for r in rs)==4
        assert sum(r['pool']==1 for r in rs)==4

def test_injected_mechanisms_match_labels_and_hold_is_not_a_label():
    for c in training_cases():
        with patch('microservice.probe_range.live.reset') as reset,patch('microservice.probe_range.live.request') as request:
            configure_training(c);reset.assert_called_once()
            ledger=request.call_args_list[0].args[2]
            assert ledger['db_hold']==c['hold']
            assert ('pool_size' in ledger)==(c['fault'] in (2,5,6,7))
            assert ('delay_seconds' in ledger)==(c['fault'] in (3,4,6,7))
            assert (request.call_count==2)==(c['fault'] in (1,4,5,7))
