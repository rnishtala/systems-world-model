"""Render the completed development screen without upgrading its claims."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-stopping-v1'
def main():
    s=json.loads((O/'summary.json').read_text());m=s['methods']
    lines=['# Frozen Dreamer stopping development results','',s['scope'],'',
        '| Method | Exact diagnoses | Mean windows | Mean ingress requests | Full-eight accuracy | Calibration failures |',
        '|---|---:|---:|---:|---:|---:|']
    for name,x in m.items():
        a=x['stopping'];lines.append(f"| {name} | {a['correct']}/64 | {a['mean_windows']:.2f} | {a['mean_requests']:.2f} | {x['fixed_prefixes']['8']['correct']}/64 | {x['calibration_failures']} |")
    lines+=['',f"Preregistered development gate: **{'passed' if s['development_gate'] else 'failed'}**.",'',
        'Costs are retrospective prefix estimates on an unchanged measured schedule. Warmup is excluded. No live early-stopping latency or alternative-action planning is demonstrated.',
        '', 'The model was frozen throughout. No new faults were injected and no model was promoted. Only training incidents were used; previous test9202 was not used for selection.',
        '', 'Nested readout/stopping selection protects outer-fold labels, but the world model previously trained on all64 trajectories. Treat these as optimistic development results, not fresh generalization evidence.', '',
        'Next: freeze a fresh shadow confirmation before making an efficiency claim.' if s['development_gate'] else 'Next: do not spend a fresh confirmation run on this stopping rule. Investigate whether earlier states encode the fault mechanisms, and improve the training objective or probe design before another bounded experiment.', '',
        'Evidence: reports/evidence/probe-stopping-v1. Protocol: reports/PROBE_STOPPING_PROTOCOL.md.']
    (R/'reports/PROBE_STOPPING_RESULTS.md').write_text('\n'.join(lines)+'\n'+'\n## Interpretation and next candidate\n\nThe stopping rule saves 0.516 of eight windows (6.4%) and 3.047 of48 ingress requests (6.3%). It fails the required one-window saving, despite64/64 development diagnoses.\n\nA post-hoc candidate is the fixed six-window Dreamer readout:64/64 in these same outer folds, with36 instead of48 measured ingress requests. A fixed six-window budget would save25% of measured requests/windows, excluding warmup. This was discovered after the screen and is not a validated saving; freeze it separately and collect fresh confirmation before claiming it.\n\nTelemetry controls failed all four inner calibration gates and therefore fell back to eight windows. Their weak grouped-CV results do not establish Dreamer superiority: world-model weights saw all development trajectories, and leaving out repeats shifts load/order schedules. The earlier fresh test had telemetry and Dreamer tied32/32. Preserve that result; do not reinterpret this optimistic development screen as a new accuracy win.\n')
if __name__=='__main__':main()
