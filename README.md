# Systems world model

Train DreamerV3 on measured service trajectories and test whether its learned state improves root-cause diagnosis over telemetry alone.

**Start with [Training and validation](docs/TRAINING.md)** for architecture diagrams, commands, data boundaries, and acceptance criteria.

| Source | Responsibility |
|---|---|
| `microservice/collect.py`, `live.py`, `ledger_probe.py` | Collect bounded, leased-fault incidents in the isolated lab |
| `microservice/service.py` | Instrumented checkout → payment → ledger service implementation |
| `microservice/probe_target.py` | Convert observations and replay only measured probe actions |
| `microservice/configuration.py`, `train.py`, `supervision.py` | Configure and train the pinned upstream Dreamer implementation |
| `microservice/model.py`, `probe_prefix.py`, `auxiliary_audit.py` | Load frozen weights, extract states, check label isolation |
| `microservice/readout.py`, `validate.py` | Fit diagnostic heads and evaluate paired telemetry/model predictions |
| `tests/` | Observation, replay, fault-isolation and validation tests |
| `scripts/verify_cleanup.py` | Reproduce archived numerical results with the consolidated code |

The repository contains source, tests and documentation. Datasets, fitted readouts, checkpoints, runtime YAML and reports are generated artifacts; keep them in an external run directory or the lab results PVC. The collector targets only `systems-world-model-lab`; training and evaluation can run offline.

Latest completed experiment: 128 training incidents, 100,000 Dreamer steps, 64 fresh test incidents. Six-observation Dreamer scored 61/64 versus 60/64 for the prior expanded-data readout; eight-observation Dreamer scored 64/64 versus telemetry's 62/64. The six-observation acceptance gate **failed**. This is not established generalization or reliable early stopping.

Historical code and evidence remain in Git at `763e555`. See [artifact recovery](docs/TRAINING.md#artifact-recovery). Cleanup changes source only; it does not delete deployed services, datasets or PVC checkpoints.

[Upstream provenance](UPSTREAM.md)
