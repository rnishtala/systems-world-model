"""Render measured temporal experiment outcomes without promotion claims."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/lab-temporal-v1'
def main():
    s=json.loads((E/'evaluation/summary.json').read_text());v=json.loads((E/'verification.json').read_text());t=json.loads((E/'models/training-61.json').read_text())
    assert v['verified']
    lines=['# Trace-only temporal diagnosis: workload-transfer results','',
        '2026-11-18. [Frozen protocol](LAB_TEMPORAL_PROTOCOL.md).',
        '[Evidence](evidence/lab-temporal-v1/summary-placeholder).'.replace('/summary-placeholder','/evaluation/summary.json'),'',
        '## Experiment','',
        'A new upstream DreamerV3 seed 61 was trained from scratch on 80 measured training incidents. '
        'Direct CPU, pool-wait and database-duration counters were excluded from all classifier inputs. '
        'Each incident exposed four independent latency/error/load/trace windows; faults persisted while '
        'concurrent load alternated between 4 and 6. After every readout was frozen, 80 fresh incidents '
        'used alternating load 7 and 9, with the same fault-severity ranges. Both phases were balanced '
        'within each of the eight known fault families. This tests workload transfer, not unseen causes.', '',
        f"Training completed {t['steps']:,} steps in {t['seconds']/60:.1f} minutes.",
        'All methods used the same four observations. Latest-snapshot and full-history telemetry '
        'classifiers received the same linear/RBF candidate grid as the Dreamer-state classifier. '
        'Selection used training-only folds; these are not independent validation of the world model.', '',
        '## Fresh evaluation','',
        '| Method | Exact diagnoses | 95% Wilson interval | Reliability gate |',
        '|---|---:|---:|---|']
    for name in ('latent','history','latest','auxiliary','shuffled_latent'):
        m=s['methods'][name];lo,hi=v['accuracy_intervals_95'][name]
        lines.append(f"| {name} | {m['correct']}/80 | {lo:.1%}–{hi:.1%} | {'Pass' if m['gate_passed'] else 'Fail'} |")
    lines+=['',f"Exploratory added-value gate: **{'passed' if s['added_value_gate'] else 'failed'}**. It requires reliability and four more correct diagnoses than the stronger telemetry baseline.",'']
    for name,p in s['paired'].items():
        lines.append(f"Against {name}: Dreamer alone correct on {p['latent_only_correct']}, baseline alone correct on {p['baseline_only_correct']}; two-sided exact paired p={p['two_sided_exact_p']:.4g}.")
    lines+=['','| Fault family | Dreamer state | History | Latest |','|---|---:|---:|---:|']
    for f,label in enumerate(('Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','All three')):
        counts=[s['methods'][name]['families'][str(f)]['correct'] for name in ('latent','history','latest')]
        lines.append(f'| {label} | {counts[0]}/10 | {counts[1]}/10 | {counts[2]}/10 |')
    clips=v['clipping_above_upper_bound']
    lines+=['', f"Checkout latency exceeded the fixed 600 ms normalization ceiling in {clips['train']['0']}/320 training windows and {clips['test']['0']}/320 test windows. Both methods received the same capped values. Ledger and downstream trace timing clipped in {clips['test']['8']} and {clips['test']['9']} test windows respectively. This is a measured information-loss limitation, not proof that it caused all diagnosis errors."]
    lines+=['','## Interpretation and next step','',
        'The new experiment did not meet the reliability gate. A small aggregate lead is not '
        'sufficient evidence of a useful world-model advantage; per-family errors and the paired '
        'comparison above are the relevant limits. This harder task changes the observation view '
        'and test workload relative to earlier experiments, so their accuracy numbers are not '
        'directly comparable.', '',
        'Next, revise the latency scaling using development data, broaden training workload coverage, and select readouts using development splits '
        'separated by operating regime, then evaluate on newly collected held-out sequences. '
        'The current random incident folds cannot measure extrapolation to a wholly different '
        'load range. Preserve this test batch as evaluation-only. There is no evidence here '
        'that a larger or more complex model is the necessary next change.']
    lines+=['','## Limits and operational outcome','',
        'One model seed and 80 controlled test incidents cannot establish general superiority. '
        'No action-selection, counterfactual-remediation, or early-stopping benefit was tested. '
        'The raw history baseline controls for simply retaining past measurements. '
        'Fault labels were excluded from Dreamer encoder/decoder, with an actual upstream policy-invariance audit. '
        'Local verification reproduced frozen readouts, test predictions and collection boundaries; '
        'latent features were not independently regenerated locally.', '',
        'Checkpoint and readouts remained unchanged during evaluation. The previous inference template '
        'and replica count were restored, and the temporary Job/stager were removed. No model was promoted. '
        'The Kubernetes job completed in 26m28s. Bulk export subsequently failed; a temporary read-only '
        'exporter recovered all evidence in 32 KiB chunks and checked file hashes. This did not rerun '
        'training or evaluation. Future runs use the bounded exporter by default.', '']
    (R/'reports/LAB_TEMPORAL_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
