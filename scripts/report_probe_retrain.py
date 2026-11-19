"""Render paired diagnostic results without claiming unmeasured generalization."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-retrain-v1'
def main():
    s=json.loads((O/'evaluation/summary.json').read_text());t=json.loads((O/'models/training-101.json').read_text())
    names=dict(latent6='New Dreamer101, six windows',latent8='New Dreamer101, eight windows',history6='Telemetry, six windows',history8='Telemetry, eight windows',old_latent6='Dreamer91, original readout',refit_latent6='Dreamer91, expanded-data readout')
    lines=['# Dreamer trained on expanded incident coverage','',
        f"Trained a fresh upstream Dreamer checkpoint (seed 101, 100,000 steps) in {t['seconds']/60:.1f} minutes on 128 designated training incidents. World-model weights changed. The original 64 and expanded-range 64 training incidents were reused without additional collection; all completed test datasets were excluded.",'',
        'Frozen six/eight-window diagnostic readouts were fitted using training-only selection. A fresh 64-incident test (seed 9503) evaluated the new checkpoint, the original Dreamer91 readout, its expanded-data refit, and matched telemetry controls on identical incidents.','',
        '| Method | Correct / 64 | False pool diagnoses | CPU / pool / delay bit errors |','|---|---:|---:|---|']
    for n,m in s['methods'].items():lines.append(f"| {names[n]} | {m['correct']} | {s['false_pool'][n]} | {s['bit_errors'][n]} |")
    lines+=['','| Regime | New latent6 | Old + expanded readout | Old + original readout | Telemetry6 |','|---|---:|---:|---:|---:|']
    for g,m in s['regimes'].items():lines.append(f"| {'Original' if g=='0' else 'Higher load/severity'} | {m['latent6']['correct']}/32 | {m['refit_latent6']['correct']}/32 | {m['old_latent6']['correct']}/32 | {m['history6']['correct']}/32 |")
    lines+=['',f"Preregistered engineering gate: **{'passed' if s['feasibility_gate'] else 'failed'}**.",
        f"Reliability: {s['reliable']}; per-regime requirements: {s['regime_gate']}; improvement over expanded-data Dreamer91 readout: {s['improvement_gate']}; matched controls: {s['matched_controls']}.",'',
        '| Paired comparator | New-only correct | Comparator-only correct | Unadjusted exact p |','|---|---:|---:|---:|']
    for n,p in s['paired'].items():lines.append(f"| {names[n]} | {p['new_only_correct']} | {p['control_only_correct']} | {p['p']:.4f} |")
    delta=s['methods']['latent6']['correct']-s['methods']['refit_latent6']['correct']
    lines+=['','## Interpretation','',
        f"The new six-window model changed exact diagnoses by {delta:+d}/64 relative to the expanded-data readout on the old world model. False pool predictions changed from {s['false_pool']['refit_latent6']} to {s['false_pool']['latent6']}. Use this same-test comparison rather than comparing scores from different incident batches.",'',
        'This is one training seed and one fresh incident batch on the same service architecture and known fault families. Dataset size and model initialization both differ from the older world model, so this does not isolate a causal data-volume effect. Readout selection remains supervised. Paired p-values are exploratory and unadjusted across comparisons; engineering thresholds do not establish statistical superiority. No held-out architecture, unseen fault family, forecast quality improvement or end-to-end resolution claim is measured here.','',
        'All eight windows were collected and six-window diagnosis was evaluated from the prefix. Actual early stopping and new latency savings were not measured. Labels are excluded from encoder/decoder inputs and verified by a runtime label-invariance audit. The action contract still represents a fixed measured probe schedule, not unrestricted counterfactual planning.','',
        'Evidence: reports/evidence/probe-retrain-v1. Protocol: reports/PROBE_RETRAIN_PROTOCOL.md. Reproduction: scripts/verify_probe_retrain.py. Checkpoint: /results/probe-retrain-v1/models/final-101 on the lab results PVC. Serving model remains unchanged; temporary resources are cleaned up and inference is restored.']
    lines+=['','## Decision and next experiment','',
        'The six-window improvement gate failed: only one extra correct diagnosis over the expanded-data baseline, two false pool diagnoses remain, shifted CPU+delay is only 2/4 correct, and six windows trail eight by three cases. Do not promote this checkpoint or claim reliable early stopping.', '',
        'All three six-window mistakes concern the pool mechanism. Eight windows correct all three on this batch, while eight-window telemetry scores 62/64. This suggests additional observations remain useful, but a perfect 64-case result is not a general reliability guarantee. The old model was compared at six windows only, so this run does not establish a full-budget improvement over the old model.', '',
        'Next, test a training-calibrated selective stopping policy: stop at six windows only when training-only confidence/consistency checks justify it; otherwise acquire the remaining two windows. Freeze the policy before a new matched actual-budget test against always-eight Dreamer and telemetry baselines. Require reliability per fault family/regime and meaningful actual request/time savings, not just pooled accuracy. Do not choose a confidence threshold using this completed test. If training validation cannot identify a useful stopping policy, retain all eight observations rather than launch another unchanged training run.']
    (R/'reports/PROBE_RETRAIN_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
