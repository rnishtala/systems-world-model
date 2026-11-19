import json
from pathlib import Path
import numpy as np
from microservice.budget_features import budget_vector,planned_action
from microservice.probe_target import frames,vector
from microservice.probe_live_budget import schedule

def test_truncated_vector_matches_frozen_contract_without_future_frames():
    p=Path(__file__).resolve().parents[1]/'reports/evidence/probe-six-v1/train.jsonl'
    for old in [json.loads(l) for l in p.read_text().splitlines()]:
        fs=frames(old);row=dict(frames=fs[:6],schedule=[[f['target'],f['requests']] for f in fs])
        assert len(row['frames'])==6
        for t in range(6):np.testing.assert_array_equal(budget_vector(row,t),vector(old,t))
        assert planned_action(row,5).shape==(2,)

def test_schedules_balance_cost_and_orders():
    for g in (0,1):
        for rep in range(4):
            plan=schedule(g,rep)
            assert len(plan)==8
            assert sum(n for _,n in plan[:6])== (36 if g==0 else 42)
            assert sum(n for _,n in plan)== (48 if g==0 else 56)
