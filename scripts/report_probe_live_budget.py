"""Report actual acquisition savings separately from diagnostic reliability."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-live-budget-v1'
def main():
    s=json.loads((O/'evaluation/summary.json').read_text());rows=[json.loads(l) for l in (O/'evaluation/incidents.jsonl').read_text().splitlines()]
    lines=['# Actual six-versus-eight observation experiment','',
        'Frozen model91 and byte-identical readouts from the preceding confirmation.64 matched scenarios, each run at both budgets:128 actual incidents. Half retain original conditions; half raise loads3/9 to4/10 and increase CPU/delay severity. Arm order balanced; the same fault seed is replayed within each pair.', '',
        '| Method | Exact diagnoses |', '|---|---:|']
    for n,m in s['methods'].items():lines.append(f"| {n} | {m['correct']}/64 |")
    lines+=['', '| Six-window Dreamer regime | Exact diagnoses |','|---|---:|']
    for g,m in s['regimes'].items():lines.append(f"| {'Original' if g=='0' else 'Higher load/severity'} | {m['correct']}/32 |")
    lines+=['',f"Preregistered combined feasibility gate: **{'passed' if s['feasibility_gate'] else 'failed'}**.",
        f"Reliability: {s['reliable']}; each-regime gate: {s['regime_gate']}; within two of full controls: {s['within_two_of_full_controls']}; efficiency: {s['efficiency_gate']}.", '',
        '| Mean cost | Six | Eight | Reduction |', '|---|---:|---:|---:|']
    for k in s['mean_costs']['6']:
        lines.append(f"| {k} | {s['mean_costs']['6'][k]:.3f} | {s['mean_costs']['8'][k]:.3f} | {100*s['fraction_saved'][k]:.1f}% |")
    lines+=['', '| Compared with | Six-window Dreamer only correct | Control only correct | Paired p |', '|---|---:|---:|---:|']
    for n,p in s['paired'].items():lines.append(f"| {n} | {p['latent6_only_correct']} | {p['control_only_correct']} | {p['p']:.4f} |")
    lines+=['', '## Scope and limitations', '',
        'Six-window arms actually stop acquisition at six. No unseen measurements are fabricated. Time measurements include lab request/metrics overhead; warmup-inclusive time starts before fault configuration and excludes the final cleanup reset. Model-inclusive time adds the frozen Dreamer extraction and all three readout computations; graph compilation is primed before collection. These are experimental pipeline timings, not production latency guarantees.', '',
        'Paired arms share fault controls but execute separately, so their noise and machine state differ. The same topology and fault families remain; only load/severity shifts. Summary-history load bins were frozen for3/9 and remain empty at4/10, although raw history retains actual values. Plain history is the important control under this shift.', '',
        'No weights were trained, no serving model promoted, and no other namespace modified. The engineering gate does not establish statistical noninferiority or broad reliability. Report all failures; do not tune against this test set.', '',
        'Protocol: reports/PROBE_LIVE_BUDGET_PROTOCOL.md. Evidence: reports/evidence/probe-live-budget-v1.']
    from microservice.mechanisms import FAULT_MASK
    errors=[dict(pair=r['pair'],regime=r['regime'],fault=r['fault'],repeat=r['repeat'],truth=FAULT_MASK[r['fault']],predicted=r['predictions']['latent'],wrong_bits=FAULT_MASK[r['fault']]^r['predictions']['latent']) for r in rows if r['budget']==6 and r['predictions']['latent']!=FAULT_MASK[r['fault']]]
    (O/'error-analysis.json').write_text(json.dumps(dict(six_window_errors=errors,all_false_pool=all(e['wrong_bits']==2 and not e['truth']&2 and e['predicted']&2 for e in errors)),indent=2)+'\n')
    lines+=['', '## Interpretation and next experiment', '',
        'Measured acquisition fell from5.74s to4.41s (23.1%); including warmup/configuration,9.75s to8.33s (14.6%). Measured requests fell52 to39 on average (25%); including warmup,88 to75 (14.8%). These savings are observed, not shadow estimates.', '',
        'Six-window Dreamer scored60/64 (93.75%), versus61/64 at eight and51/64 for plain history at six. Its familiar-regime score remained32/32, but shifted performance fell to28/32. The gate failed because the CPU-plus-delay family scored only1/4 under the shift (5/8 overall). All four six-window errors falsely added pool contention; three were CPU-plus-delay, one delay-only. Eight observations recovered only one of these cases.', '',
        'The plain-history comparison has unadjusted paired p=0.0225, but several controls were compared, and the load-bin summary control is structurally disadvantaged under the shift. Treat this as exploratory evidence of a representation benefit, not broad superiority.', '',
        'Next: collect a separate factorial training set that varies load, CPU strength, delay strength, DB hold, and pool size independently with overlapping latency ranges. Include both true and absent pool contention at high delay. Preserve this completed test as held-out audit evidence; do not train on these incidents. Retrain and calibrate on new training/development seeds, freeze again, then evaluate on another untouched shifted set. Keep six observations as the candidate budget; do not promote the current model.', '',
        'The Kubernetes job took19m44s. Two transient SSH monitoring interruptions did not restart or repeat collection. All128 arms completed within the35s safety bound (maximum15.08s); final health verification confirms serving restoration and no active experiment jobs.']
    (R/'reports/PROBE_LIVE_BUDGET_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
