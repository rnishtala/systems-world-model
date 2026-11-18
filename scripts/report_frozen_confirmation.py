"""Report the fresh confirmation without retuning or changing its primary."""
import json
from pathlib import Path
from scripts.verify_lab_temporal import wilson
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/lab-confirm-v1'
NAMES={'latent':'Dreamer latent state (primary)','response':'Load response','history':'Full history','response_history':'Load response + history','response_latent':'Load response + latent state','response_forecast':'Load response + forecast features','auxiliary':'Dreamer auxiliary head'}
def main():
    s=json.loads((E/'evaluation/summary.json').read_text());v=json.loads((E/'verification.json').read_text());assert v['verified']
    lines=['# Fresh confirmation with frozen readouts','','2026-11-18. [Protocol](FROZEN_CONFIRMATION_PROTOCOL.md) · [Evidence](evidence/lab-confirm-v1/evaluation/summary.json).','',
        'The unchanged model-81 checkpoint and all readouts were frozen before 64 fresh incidents (seed 9002) were collected. Readout configuration selection and fitting used only the original 128 training incidents. No model training or validation-time tuning occurred.', '',
        'This is a new confirmation motivated by development findings, not a reinterpretation of the failed forecast-feature development gate. The primary was fixed to the latent-state readout before collection.','',
        '| Method | Exact diagnoses | 95% Wilson interval |','|---|---:|---:|']
    for name,m in s['methods'].items():
        lo,hi=wilson(m['correct'],m['n']);lines.append(f"| {NAMES[name]} | {m['correct']}/{m['n']} | {lo:.1%}–{hi:.1%} |")
    lines+=['','| Criterion | Result |','|---|---|']
    for key,label in [('primary_reliable','Primary diagnosis reliability'),('added_value','At least four more correct than strongest telemetry control'),('forecast_gate','Forecast MSE below persistence'),('combined_gate','Combined gate')]:
        lines.append(f"| {label} | {'Pass' if s[key] else 'Fail'} |")
    lines+=['','Reliability requires >=58/64 overall, healthy 8/8 and every other family >=6/8. These are prespecified sample thresholds, not confidence bounds on production reliability.','',
        '| Family | Dreamer primary | Load response | Forecast-feature readout |','|---|---:|---:|---:|']
    for i,name in enumerate(('Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','All three')):
        a=s['methods'];lines.append('| '+name+' | '+' | '.join(str(a[k]['families'][str(i)]['correct'])+'/8' for k in ('latent','response','response_forecast'))+' |')
    f=s['forecast_mse'];reduction=1-f['predicted']/f['persistence']
    lines+=['','## Forecasts and paired diagnosis comparisons','',
        f"Corrected forecast MSE: {f['predicted']:.6f}; persistence: {f['persistence']:.6f}. Relative error reduction: {reduction:.1%}. Units are the experiment's log-latency observations, not seconds."]
    for name,p in s['paired'].items():
        lines.append(f"- Against {NAMES[name]}: primary-only correct {p['latent_only_correct']}, baseline-only correct {p['baseline_only_correct']}; two-sided exact McNemar p={p['two_sided_exact_p']:.4g}.")
    lines+=['','These are paired comparisons on one fresh batch. Control comparisons are not corrected for multiplicity; do not turn sample differences into a broad superiority claim.','',
        '## Interpretation','']
    if s['combined_gate']:
        lines+=['The prespecified combined criterion passed on this batch. Replication on additional workload regimes and model seeds remains necessary before claiming dependable operational triage. No model was promoted.']
    else:
        lines+=['The combined criterion failed. This experiment does not establish reliable diagnostic superiority from the world model. Do not change the primary to whichever secondary readout scored best after observing this batch. No further fault collection or training was launched.']
    lines+=['',
        'The forecast-feature control scored 59/64, only one more correct than load response, and identified CPU-plus-pool faults in only 5/8 cases. It does not rescue the failed primary result or meet the per-family minimum. The primary missed pool-only (5/8), CPU-plus-delay (4/8), and CPU-plus-pool (5/8) cases. A next experiment should target these diagnostic ambiguities rather than repeat broad training without a specific hypothesis.', '']
    lines+=['','## Verification and limits','',
        'Independent verification reproduced training-only configuration selection and readout fits, checked freeze-before-collection timing, workload balancing and measurement validity, reproduced predictions and metrics, and verified export hashes. The job checked checkpoint and readout immutability, label isolation and workload-action alignment. Latent states and forecasts were not regenerated locally.','',
        'The world model and readouts did not train on these incidents. Fault classes remain the same eight controlled families, all measured requests succeeded, and only one checkpoint and batch were assessed. Earlier validation influenced the research direction, but no confirmation data influenced this frozen fit. This does not establish unseen-fault diagnosis, autonomous investigation or remediation.','',
        'Work remained inside the systems-world-model repository and namespace. Inference was restored to its original template and replica count; temporary resources were removed.','']
    (R/'reports/FROZEN_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
