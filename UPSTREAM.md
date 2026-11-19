# DreamerV3 provenance

Project: https://danijar.com/project/dreamerv3/
Code: https://github.com/danijar/dreamerv3
Pinned commit: `e01491fad6434b2245a3b8ca201dd7faedcc458c`
License: MIT (retained in every experiment bundle).

The upstream repository describes itself as a reimplementation of DreamerV3,
unrelated to Google/DeepMind as a supported software product. This project
reuses that code rather than claiming an official Google distribution.

The upstream `dreamerv3/` and `embodied/` files are unmodified. Our runner supplies
an environment factory to the upstream main module and invokes its training loop.
The encoder, RSSM, prediction heads, replay, imagination, actor and critic remain
upstream implementations. Our subclass adds a supervised mechanism head and loss,
while excluding labels from encoder, decoder and policy inputs. Numeric observations and configuration overrides tailor
the environment and compute budget; we do not substitute a custom learned model.

Our results are a small diagnostic-task experiment, not a reproduction of the
published Dreamer benchmark suite. Weights are trained from scratch. No LLM is
trained or called in this experiment.
