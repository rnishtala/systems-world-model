import numpy as np
import pytest

from microservice.action_balanced import schedule
from microservice.action_masking import folds, mask_errors
from microservice.action_model import WIDTH
from tests.test_action_model import row


def test_mask_uses_validity_even_for_measured_zero():
    target = np.zeros((1, 2, WIDTH))
    target[0, 0, -2] = target[0, 1, -1] = 1
    result = mask_errors(np.ones_like(target), target, np)
    assert result[0, 0].sum() == 53
    assert result[0, 1].sum() == 6
    assert (result[0, 0, -6:-2] == 0).all()
    assert (result[0, 1, :-6] == 0).all()
    assert (result[..., -2:] == 1).all()


def test_whole_profile_folds_cover_each_regime_and_action():
    rows = [dict(row(action=c['action']), group=c['group']) for c in schedule(113) if c['split']=='train']
    seen = set()
    for train, dev in folds(rows):
        held = {r['group'] for r in dev}
        assert not seen.intersection(held)
        assert not held.intersection(r['group'] for r in train)
        assert len({g.rsplit('-', 1)[0] for g in held}) == 8
        assert len(train) == len(dev) == 32
        seen.update(held)
    assert len(seen) == 16
    with pytest.raises(ValueError):
        list(folds(rows[:-1]))
