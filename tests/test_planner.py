import numpy as np
from experiment.planner import combine_values,choose_action


def test_expectation_precedes_diagnosis_selection_and_terminal_stops():
    first=np.array([[-.2],[1.95]])
    continuation=np.ones((2,1))
    last=np.array([[[[2.,-2.],[1.,1.],[-2.,-2.]]],[[[100.,100.],[100.,100.],[100.,100.]]]])
    scores=combine_values(first,continuation,last,np.array([True,False]))
    np.testing.assert_allclose(scores,[.8,1.95])


def test_bounded_policy_masks_only_already_used_information_actions():
    vector=np.zeros(9);vector[6:8]=1
    assert choose_action(np.array([100.,99.,98.,1.,2.,3.,0.]),vector,True)==5
