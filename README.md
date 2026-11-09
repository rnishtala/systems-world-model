# Systems World Model

An independent research MVP that reuses DreamerV3's implementation to learn
system triage: which diagnostic check to perform and when to make a diagnosis.
Train on our own simulator and system data, rather than reuse game-trained weights.

A later extension pairs the trained agent with a frozen LLM, reviewed execution,
and independent recovery verification for end-to-end resolution in an isolated lab.

**Status:** planning only. No Dreamer dependency installed, model trained,
application implemented, or cluster resources deployed.

Read [MVP_PLAN.md](MVP_PLAN.md) for the architecture walkthrough, smallest experiment,
LLM resolution flow, evaluation, and proposed independent AMD deployment.

This is not an AutoSRE component or a production remediation agent.
