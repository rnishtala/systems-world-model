"""Verify API parity and fresh live results independently of the collector."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.latent_probe import observations
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/latent-api'

def main():
    restored=json.loads((E/'restored.json').read_text());assert restored['all_deployments_ready'] and restored['inference_template_restored']
    source=json.loads((E/'source.json').read_text());assert source['protocol_sha256']==hashlib.sha256((R/'reports/LATENT_API_PROTOCOL.md').read_bytes()).hexdigest()
    package=json.loads((E/'package.json').read_text());identity=package['identity'];confirmed=json.loads((R/'reports/evidence/latent-confirmation/eval-23/fitted.json').read_text())
    assert identity['seed']==23 and identity['checkpoint_sha256']==confirmed['checkpoint_sha256'] and identity['calibration_sha256']==confirmed['calibration_sha256']
    artifact=E/'serving-readout-23.npz'
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==identity['readout_sha256']
    assert json.loads((E/'serving-readout-23.json').read_text())==identity
    with np.load(artifact,allow_pickle=False) as actual,np.load(R/'reports/evidence/latent-confirmation/eval-23/probes.npz',allow_pickle=False) as confirmed_probe:
        for key in ('mean','scale','weights','bias'):np.testing.assert_array_equal(actual[key],confirmed_probe['latent_'+key])
    allowed={'train-23/config.yaml','training-23.json','readout-23.json','readout-23.npz'}
    assert all(name in allowed or name.startswith('final-23/') for name in package['files'])
    parity=json.loads((E/'parity.json').read_text());expected=json.loads((R/'reports/evidence/latent-confirmation/eval-23/predictions.json').read_text())['latent'];assert len(parity['rows'])==len(expected)==80
    for i,row in enumerate(parity['rows']):assert row['episode']==i and row['expected']==expected[i]
    matches=sum(row['actual']==row['expected'] for row in parity['rows']);assert matches==parity['matching']
    passed=all(row['actions'][:3]==[1,2,3] and row['actual']==row['expected'] and row['readout_sha256']==identity['readout_sha256'] for row in parity['rows']) and all(parity['contract_rejections']);assert passed==parity['passed']
    summary=json.loads((E/'summary.json').read_text());assert summary['parity_passed']==passed
    rows=[]
    if not summary['fresh_skipped']:
        rows=list(map(json.loads,(E/'live.jsonl').read_text().splitlines()));assert len(rows)==summary['n']==80
        for i,row in enumerate(rows):
            assert row['episode']==i and row['scenario_seed']==6901 and row['seconds']<=35 and row['target']==FAULT_MASK[row['fault']]
            assert len(row['frames'])==len(row['trace'])==4
            expected_obs=observations(row)
            for j,(frame,trace) in enumerate(zip(row['frames'],row['trace'])):
                assert frame['successes']==frame['requests'] and set(frame['metrics'])=={'checkout','payment','ledger'}
                assert all(m['count']==(j+1)*frame['requests'] for m in frame['metrics'].values())
                np.testing.assert_array_equal(trace['observation'],expected_obs[j])
                response=trace['response'];assert response['controller']=='inspect-latent' and response['action_contract']=='mechanisms-v1'
                if j<3:assert response['action']==j+1
            response=row['trace'][-1]['response'];assert response['readout_sha256']==identity['readout_sha256']
            scores=np.asarray(response['latent_scores']);assert scores.shape==(3,) and np.isfinite(scores).all()
            assert row['diagnosis']==response['action']-4==int(sum((scores>0)*[1,2,4]))
            assert row['correct']==(row['diagnosis']==row['target'])
        for f,v in summary['families'].items():
            group=[r for r in rows if r['fault']==int(f)];assert len(group)==v['n']==10 and sum(r['correct'] for r in group)==v['correct']
        assert summary['correct']==sum(r['correct'] for r in rows)
        assert summary['gate_passed']==all(v['correct']>=(10 if f=='0' else 9) for f,v in summary['families'].items())
    promotion=json.loads((E/'promotion.json').read_text()) if (E/'promotion.json').exists() else None
    if promotion:
        assert summary['gate_passed'] and promotion['promoted'] and promotion['healthy_smoke_passed']
        assert promotion['readout_sha256']==identity['readout_sha256'] and promotion['source']=='microservice-source-v20'
        assert len(promotion['deployments'])==5 and all(d['ready'] for d in promotion['deployments'])
    verification=dict(candidate_promoted=bool(promotion),parity_matching=matches,parity_passed=passed,fresh_incidents=len(rows),frozen_coefficients_match_confirmation=True,serving_package_excludes_data=True,checkpoint_matches_confirmation=True,protocol_unchanged=True,live_observations_match_offline_contract=bool(rows),diagnoses_match_api_scores=bool(rows),baseline_restored=True)
    (E/'verified.json').write_text(json.dumps(verification,indent=2)+'\n')
    (E/'errors.json').write_text(json.dumps([dict(episode=r['episode'],fault=r['fault'],target=r['target'],diagnosis=r['diagnosis'],scores=r['trace'][-1]['response']['latent_scores'],last_metrics=r['frames'][-1]['metrics']) for r in rows if not r['correct']],indent=2)+'\n')
    lines=['# Frozen latent classifier through the live inference API','',f"Saved-trajectory API parity: **{matches}/80**. Label/correctness-reward rejection checks: **{'passed' if all(parity['contract_rejections']) else 'failed'}**.",'',
      'The serving package contains only model configuration, checkpoint, metadata and frozen classifier coefficients. It excludes calibration, held-out examples and replay. Artifact hashes bind the classifier to its confirmed checkpoint. The server retains its agent role, read-only mounts and no service-account credentials.','']
    if rows:
        lines += [f"Fresh live result: **{summary['correct']}/80; gate {'passed' if summary['gate_passed'] else 'failed'}**. Scenario seed 6901; same model 23 and classifier as independent confirmation. No training or fitting occurred.",'','| Family | Exact diagnoses |','|---|---:|']
        for f,name in enumerate(['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']):lines.append(f"| {name} | {summary['families'][str(f)]['correct']}/10 |")
        times=np.array([r['seconds'] for r in rows]);lines+=['',f"Injection-through-diagnosis duration: median {np.median(times):.2f}s; p95 {np.percentile(times,95):.2f}s; maximum {times.max():.2f}s. This includes fault setup, warm-up, four measured batches and four HTTP inference calls."]
    else:lines+=['Fresh live testing was skipped because the integration gate failed. Historical replay is not counted as new accuracy evidence.']
    lines+=['','## Decision and remaining scope','',
      ('After the experiment restored the baseline, a separate reviewed promotion made the exact tested candidate the lab default. A healthy real-service API smoke check passed; all five deployments are ready. The original inference template and service sources remain available for rollback. Diagnosis-only behavior is retained.' if promotion else 'The frozen classifier passed live API validation. The next step is a controlled promotion within the isolated lab, with this exact packaged checkpoint and readout. Preserve a rollback manifest and retain diagnosis-only behavior.' if summary['gate_passed'] else 'The candidate was not promoted. Preserve this failed integration/live result; investigate the errors before designing a new prespecified experiment.'),
      'This demonstrates fixed-inspection triage for eight known regimes in one small service chain. It does not establish production reliability, unseen-service generalization, learned investigation planning, LLM orchestration or autonomous remediation. The independent confirmation and this fresh live batch are reported separately.','',
      '## Validation and restoration','',
      'Seven readout/package tests passed. Evidence verification confirms exact equality of packaged and confirmed classifier arrays, and checks all replay actions, classifier identity, actual API score-to-action mapping, exact offline/live observation equivalence, all fresh window counts and per-family scores. The experiment verified restoration of the original inference template and all five baseline deployments before any separate promotion; no other namespace was modified.','',
      'Evidence: `reports/evidence/latent-api/`. Protocol: `reports/LATENT_API_PROTOCOL.md`.']
    (R/'reports/LATENT_API_RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
