"""Report measured regime validation, explicitly distinguishing any final test."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/lab-regime-v1'
NAMES={'latent':'Dreamer state + readout','history':'Full-history telemetry','latest':'Latest snapshot','auxiliary':'Dreamer auxiliary head','shuffled_latent':'Shuffled-label control'}
def main():
    s=json.loads((E/'evaluation/summary.json').read_text());v=json.loads((E/'verification.json').read_text());training=json.loads((E/'models/training-71.json').read_text())
    assert v['verified']
    lines=['# Uncapped trace diagnosis with workload-regime validation','',
           '2026-11-18. [Protocol](LAB_REGIME_PROTOCOL.md) · [Evidence](evidence/lab-regime-v1/evaluation/summary.json).','',
           '## Design','',
           'A new DreamerV3 seed 71 trained on 128 fresh incidents across four load regimes. '
           'Latency inputs use log1p(seconds / 0.1), with no hard cap. CPU, pool-wait and database '
           'duration counters remain hidden from all methods. Architecture and training budget are unchanged.', '',
           f"Training completed {training['steps']:,} steps in {training['seconds']/60:.1f} minutes.",
           'Training used alternating loads 2/4, 4/6, 6/8 and 8/10. Validation used 64 new incidents '
           'at 3/7 and 5/9, absent from world-model and readout fitting. Each readout candidate was '
           'fitted on training data only; validation selected an existing fit without refitting. '
           'The final test, if warranted, uses new mixed sequences 3/11/6/9 and reverse.', '',
           '## Validation used for readout selection','',
           '| Method | Exact diagnoses |','|---|---:|']
    for name,m in s['validation']['methods'].items():lines.append(f"| {NAMES[name]} | {m['correct']}/{m['n']} |")
    lines+=['',f"Preregistered screening gate: **{'passed' if s['validation_gate'] else 'failed'}**. Requires Dreamer >=58/64, all eight healthy correct, >=6/8 in every other family, and no more than two fewer correct than the stronger telemetry baseline.", '',
            'These figures selected the readouts, so they are not an independent final-test estimate. '
            'The world model itself did not train on validation incidents. Validation paired statistics '
            'in the evidence are descriptive after selection, not confirmatory significance tests.','',
            '| Fault family | Dreamer state | History | Latest |','|---|---:|---:|---:|']
    for f,label in enumerate(('Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','All three')):
        m=s['validation']['methods'];a=[m[n]['families'][str(f)]['correct'] for n in ('latent','history','latest')]
        lines.append(f'| {label} | {a[0]}/8 | {a[1]}/8 | {a[2]}/8 |')
    if s['final_test_collected']:
        lines+=['','## Fresh final test','','| Method | Exact diagnoses | 95% Wilson interval |','|---|---:|---:|']
        for name,m in s['test']['methods'].items():
            lo,hi=v['accuracy_intervals_95']['test'][name];lines.append(f"| {NAMES[name]} | {m['correct']}/80 | {lo:.1%}–{hi:.1%} |")
        lines+=['',f"Final added-value gate: **{'passed' if s['added_value_gate'] else 'failed'}**."]
        for name,p in s['test']['paired'].items():lines.append(f"Against {name}: Dreamer-only wins {p['latent_only_correct']}, baseline-only wins {p['baseline_only_correct']}, nominal paired exact p={p['two_sided_exact_p']:.4g}.")
    else:lines+=['','## Final test deferred','','The screening gate failed. No final-test incidents were collected, saving 80 live fault injections. No final-test accuracy or generalization advantage is claimed.']
    lines+=['','## Verification and limits','',
            'Independent verification reproduced training-only readouts, validation selection and saved '
            'predictions; checked measured observations, workload balancing, split timing, hashes and '
            'wait-action alignment. The upstream checkpoint passed label-invariance checks. The prior '
            'inference template and replica count were restored; temporary resources were removed. '
            'No model was promoted. Latent features were not regenerated locally.', '',
            'This is one model seed, eight known controlled fault families, and fixed observation budgets. '
            'It does not establish autonomous investigation, unseen-fault diagnosis, remediation, or a '
            'unique benefit from generative world-model training over other recurrent representations.', '']
    (R/'reports/LAB_REGIME_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
