"""Render the prespecified pilot result without selecting a new primary."""
import json
from scripts.evaluate_ledger_probe import O,R

def main():
    s=json.loads((O/'summary.json').read_text());m=s['methods'];p=s['paired']
    lines=['# Direct-ledger acquisition pilot','',
      '2026-11-18. [Protocol](LEDGER_PROBE_PROTOCOL.md) · [Evidence](evidence/ledger-probe-v1/summary.json).','',
      'The prespecified screening gate '+('passed' if s['screening_gate'] else 'failed')+'. This tests the value of an extra observation, not Dreamer performance. No Dreamer weights were trained or checkpoint promoted.','',
      'Thirty-two development incidents selected and fitted the classifiers. Their parameters were frozen before 32 fresh test incidents. Each split contains four cases per fault family. Both extra probes used 12 ingress requests and 12 ledger/database requests; the base checkout observations were shared.','',
      '| Test readout | Pool + delay exact (primary) | All three mechanisms exact | CPU / pool / delay bit errors |',
      '|---|---:|---:|---:|']
    for n,title in [('base','Base checkout only'),('checkout','Base + repeated checkout'),('ledger','Base + direct ledger')]:
        a=m[n];lines.append(f"| {title} | {a['pool_delay_correct']}/32 | {a['exact_correct']}/32 | "+' / '.join(map(str,a['bit_errors']))+' |')
    lines+=['','| Family | Repeated checkout: pool/delay | Direct ledger: pool/delay |','|---|---:|---:|']
    for f,label in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','All three']):
        lines.append(f"| {label} | {m['checkout']['families'][str(f)]['pool_delay_correct']}/4 | {m['ledger']['families'][str(f)]['pool_delay_correct']}/4 |")
    lines+=['',f"On the primary outcome, direct ledger was uniquely correct on {p['ledger_only_correct']} cases; repeated checkout was uniquely correct on {p['checkout_only_correct']}. Two-sided exact McNemar p={p['exact_mcnemar_p']:.4f}. This small sample does not establish production reliability.",'',
      'The screening gate requires direct ledger >=28/32, a gain of at least four over repeated checkout, healthy 4/4 and every family >=3/4. It is a research decision rule, not a statistical significance claim.','',
      '## Operational interruption','',
      'The original seed-9102 test was interrupted by SSH monitoring failures before any test data were exported or scored. Development data and frozen classifiers were retained. A documented replacement used seed 9103 with identical settings and success criteria. The recovery runner preserved partial logs and tolerated monitoring outages without restarting the job. See [recovery amendment](LEDGER_PROBE_RECOVERY.md). This replacement and the missing original test observations are part of the experiment history.','',
      '## Cost and ordering','',
      f"Fresh test incidents averaged {s['mean_incident_seconds']:.2f}s (maximum {s['max_incident_seconds']:.2f}s), including both arms and warmups. The extra checkout pair averaged {s['extra_probe_seconds']['checkout']:.2f}s and the ledger pair {s['extra_probe_seconds']['ledger']:.2f}s, including telemetry reads. Counts match, but upstream work and wall time intentionally differ.",'',
      '| First extra probe | Repeated checkout: pool/delay | Direct ledger: pool/delay |','|---|---:|---:|']
    for order in ('checkout','ledger'):
        a=m['checkout']['order'][order];b=m['ledger']['order'][order]
        lines.append(f"| {order} | {a['pool_delay_correct']}/{a['n']} | {b['pool_delay_correct']}/{b['n']} |")
    lines+=['','## Interpretation and next step','']
    if s['screening_gate']:
        lines+=['The extra probe passed the screening rule and warrants collecting trajectories with explicit probe target and concurrency actions. Train and evaluate a new Dreamer action contract on those trajectories, with an equally informed telemetry control and a separate frozen test. The existing workload-only model cannot predict this untrained intervention. This result does not yet establish that Dreamer adds value.']
    else:
        lines+=['Do not retrain Dreamer on the assumption that this probe improves triage: the matched-budget screening criterion was not met. Preserve this negative result. Further work needs a distinct, preregistered hypothesis rather than changing thresholds or selecting a different metric from this test.']
    lines+=['','## Limits and verification','',
      'The test uses the same eight synthetic fault families and 3/9 concurrency levels as development, with independently sampled severities. All requests succeeded; this is performance-fault triage, not outages or unseen-fault generalization. Both arms share an incident and execute sequentially; balanced order reduces but cannot eliminate state carryover. Declared client concurrency is not a guarantee of simultaneous arrival at the service. The control uses the same ledger-latency feature shape; additional upstream metrics from the repeated probe were not added, so this is a matched-feature acquisition comparison, not the strongest possible telemetry classifier.','',
      'Verification checked source hashes, balanced incidents, successful requests and per-service counts, fault-duration limits, equal request budgets, label isolation, training-only selection, frozen parameters before test collection, predictions and metrics. Three focused tests cover counter exclusion, measurement failure handling, and monitor-timeout recovery.','',
      'All work remained in systems-world-model-lab. The collector mounted no model or results volume and held no Kubernetes API credential. Inference was temporarily paused for quota, then restored to its original template and replica count. Temporary jobs and the collector ConfigMap were removed; final namespace health was checked.','']
    (R/'reports/LEDGER_PROBE_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
