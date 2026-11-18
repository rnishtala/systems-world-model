import copy
import json
from pathlib import Path
from scripts.freeze_order_policy import confirmation_eligible


def test_historical_final_policy_is_not_eligible():
    root = Path(__file__).resolve().parents[1]
    policies = json.loads((root/'reports/evidence/lab-order-confirm-v1/frozen/policy.json').read_text())['policies']
    assert not confirmation_eligible(policies)


def test_requires_reliability_and_half_check_advantage():
    def policy(order, checks):
        return dict(order=order, candidates=[dict(order=order, calibration_passed=True,
            stopping=dict(gate_passed=True, correct=80, mean_checks=checks), full=dict(correct=80))])
    policies = dict(latent=policy('213',2),raw=policy('321',2.5))
    assert confirmation_eligible(policies)
    for field, value in [('gate_passed',False),('correct',78),('mean_checks',2.01)]:
        bad = copy.deepcopy(policies)
        bad['latent']['candidates'][0]['stopping'][field] = value
        assert not confirmation_eligible(bad)
