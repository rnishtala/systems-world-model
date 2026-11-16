Pinned upstream `dreamerv3/agent.py:442` computes normalized action entropy as
`(mean_entropy - min_entropy) / (max_entropy - min_entropy)`.
With one possible action, both entropy bounds are zero, giving 0/0.
This explains the NaN `rand/action` logging statistic in the passive environment.
The action policy loss is zero; this experiment does not learn an action policy.
Model/readout numerical finiteness is checked separately. No upstream code changed.
