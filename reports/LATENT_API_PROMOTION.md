# Controlled lab promotion decision

The API experiment restores the original deployment as prespecified. A separate operational promotion may follow only after the completed results have been reviewed and independently verified; it does not change the experiment protocol or its acceptance gate.

Required evidence: 80/80 saved-diagnosis parity, successful contract rejection checks, a passing fresh per-family gate, verified observation/score consistency, seven passing readout/package tests, and successful restoration of the original inference template.

Promote only the exact tested source ConfigMap microservice-source-v20, model seed 23 and packaged frozen readout at micro-latent-api-v1/serving. Keep diagnosis-only behavior, fixed inspections, read-only model mounts, agent network role and the existing namespace resource budget. Update the three toy services to the same tested source for clean measurement support. No other namespace changes.

After rollout, require a healthy real-service diagnosis through the API with clean windows and the expected classifier hash. Retain deploy/microservice-agent-api-restore.json and reports/evidence/latent-api/original-sources.json as rollback inputs. Roll back on a known rollout/smoke failure; preserve state for inspection on connectivity loss rather than making blind changes.

This makes the limited known-regime POC available in the isolated lab. It does not authorize remediation, expose the service externally, or claim production reliability.
