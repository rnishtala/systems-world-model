import pytest
from microservice.coverage_collect import validate

def test_requires_complete_balanced_successful_trajectories():
    rows=[dict(fault=i,frames=[dict(successes=2,requests=2)]*4) for i in range(8)]
    validate(rows,{i:1 for i in range(8)})
    with pytest.raises(ValueError):validate(rows[:-1],{i:1 for i in range(8)})
    rows[-1]['frames']=[dict(successes=1,requests=2)]*4
    with pytest.raises(ValueError):validate(rows,{i:1 for i in range(8)})
