"""Report expanded-range diagnostic readout training, not world-model training."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-range-v1'
def main():
    s=json.loads((O/'evaluation/summary.json').read_text())
    lines=['# Expanded-range diagnostic readout experiment','',
        'Dreamer91 weights stayed frozen. Added64 newly collected training incidents to the existing64, broadened CPU/delay/DBhold/pool ranges and loads, then fitted new supervised diagnostic readouts. All readouts were frozen before collecting64 fresh test incidents. Prior completed test incidents were excluded from training.', '',
        '| Method | Correct /64 | False pool diagnoses | CPU / pool / delay bit errors |','|---|---:|---:|---|']
    for n,m in s['methods'].items():lines.append(f"| {n} | {m['correct']} | {s['false_pool'][n]} | {s['bit_errors'][n]} |")
    lines+=['', '| Regime | New latent6 | Old latent6 | History6 |','|---|---:|---:|---:|']
    for g,m in s['regimes'].items():lines.append(f"| {'Original' if g=='0' else 'Higher load/severity'} | {m['latent6']['correct']}/32 | {m['old_latent6']['correct']}/32 | {m['history6']['correct']}/32 |")
    lines+=['',f"Preregistered feasibility gate: **{'passed' if s['feasibility_gate'] else 'failed'}**.",
        f"Reliability: {s['reliable']}; regime gate: {s['regime_gate']}; improvement over old readout: {s['improvement_gate']}; matched controls: {s['matched_controls']}.", '',
        '| Control | New-only correct | Control-only correct | Unadjusted paired p |','|---|---:|---:|---:|']
    for n,p in s['paired'].items():lines.append(f"| {n} | {p['new_only_correct']} | {p['control_only_correct']} | {p['p']:.4f} |")
    lines+=['', '## Interpretation limits','',
        'Any improvement here comes from the supervised diagnostic readout and its expanded data, not retrained Dreamer weights. The updated telemetry controls receive the same expanded training data. This distinguishes readout calibration from an improved world-model representation.', '',
        'Six-observation scores use prefixes of eight-window shadow trajectories; this run makes no additional real latency-saving claim. The prior actual-budget timing evidence remains separate. The test covers new incidents on the same services and fault families, with stronger load/severity, not unseen architectures. The engineering gate is not statistical noninferiority; paired p-values are unadjusted across comparisons.', '',
        'Protocol: reports/PROBE_RANGE_PROTOCOL.md. Evidence: reports/evidence/probe-range-v1. No serving promotion or other namespace changes.']
    lines+=['', '## Conclusion and next experiment','',
        'The updated six-observation readout achieved62/64 versus61/64 for the old readout on exactly the same fresh incidents. It corrected one pool-bit mistake, but false positive pool diagnoses stayed at two. Both regimes scored31/32 for the new readout. The basic reliability gates passed, but the required two-diagnosis improvement and reduction in false pool diagnoses did not. Do not present this as a successful fix.', '',
        'The earlier60/64 score came from different incidents and is not a valid paired improvement baseline. Here the old shifted-regime readout already scored31/32, illustrating sample variability; the same-test comparison is the meaningful one.', '',
        'Next: train a fresh Dreamer world model on the128 now-available designated training trajectories, retaining the same observation/action contract and supervision. Compare its six-observation readout against both the original and this refitted frozen-model readout on another untouched test seed. Reuse the newly collected training data; do not repeat training collection, train on completed tests, or change thresholds after outcomes. The readout-only experiment did not establish that weight retraining will work, but it did not resolve the targeted error cheaply.', '',
        'No serving promotion. All source/export hashes, training-only fits, prefix telemetry, freeze chronology, scores and gates are independently verified by scripts/verify_probe_range.py. Final namespace health is recorded separately in the evidence.']
    (R/'reports/PROBE_RANGE_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
