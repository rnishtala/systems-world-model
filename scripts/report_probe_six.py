"""Report fresh fixed-six confirmation with shadow-cost limitations."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-six-v1'
def main():
    s=json.loads((O/'evaluation/summary.json').read_text());rows=[json.loads(l) for l in (O/'evaluation/test.jsonl').read_text().splitlines()]
    lines=['# Fresh six-observation confirmation','',
        'Frozen Dreamer model91 and readouts selected only on64 training incidents;32 new incidents, seed9302. Same fault families/topology and load distribution as training. No model training or serving promotion.', '',
        '| Method | Exact diagnoses | Observations |', '|---|---:|---:|']
    for n,m in s['methods'].items():lines.append(f"| {n} | {m['correct']}/32 | {n[-1]} |")
    lines+=['',f"Prespecified feasibility gate: **{'passed' if s['feasibility_gate'] else 'failed'}**.",'',
        '| Compared with | Six-window Dreamer only correct | Control only correct | Paired p |', '|---|---:|---:|---:|']
    for n,p in s['paired'].items():lines.append(f"| {n} | {p['latent6_only_correct']} | {p['control_only_correct']} | {p['p']:.4f} |")
    lines+=['', 'Six measured windows imply36 ingress requests versus48 for eight:25% less measured evidence. Including36 warmup requests, the reduction is14.3%. These are shadow estimates: this experiment collected all eight windows, so actual latency savings have not been demonstrated.', '',
        'The six-window candidate was selected from an earlier optimistic development screen. All parameters were frozen before this new collection; test labels were used only for scoring. The small32-incident sample supports an engineering feasibility assessment, not statistical noninferiority, production reliability, unseen-service generalization, or adaptive probe planning.', '',
        'The primary gate permits one fewer correct diagnosis than the strongest full-eight control. A gate pass therefore does not establish superior accuracy or a unique world-model benefit; the equally budgeted telemetry results above must be considered.', '',
        f"Collection wall time within incidents: {sum(r['seconds'] for r in rows):.1f}s total; max {max(r['seconds'] for r in rows):.2f}s. Includes warmup and eight-window measurement, excludes model initialization/extraction and infrastructure lifecycle.", '',
        'Protocol: reports/PROBE_SIX_PROTOCOL.md. Evidence: reports/evidence/probe-six-v1.']
    if s['feasibility_gate']:
        lines+=['', '## Interpretation and next step', '',
            'The six-window Dreamer readout matched full-eight accuracy on this fresh sample. Compared with equally budgeted telemetry, its two/three additional correct diagnoses yield paired p-values0.50/0.25, so this is an encouraging signal rather than established superiority.', '',
            'Next: keep these readouts frozen and run a larger, balanced comparison that actually stops collection at six windows. Measure warmup-inclusive requests and latency, and include harder severity/load shifts. Do not promote the model on this32-incident result alone.']
    (R/'reports/PROBE_SIX_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
