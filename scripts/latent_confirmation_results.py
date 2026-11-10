"""Verify frozen calibration fitting, fresh observations and independent-seed scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.latent_probe import fit_probe,predict,labels,observations
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/latent-confirmation';D=E/'eval-23'
FAMILIES=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']

def main():
    assert json.loads((E/'restored.json').read_text())['all_deployments_ready']
    source=json.loads((E/'source.json').read_text());assert source['protocol_sha256']==hashlib.sha256((R/'reports/LATENT_CONFIRMATION_PROTOCOL.md').read_bytes()).hexdigest()
    training=json.loads((E/'training-23.json').read_text());identity=json.loads((E/'identity-23.json').read_text())
    expected=hashlib.sha256((R/'reports/evidence/complete-coverage/calibration.jsonl').read_bytes()).hexdigest()
    assert training['seed']==identity['seed']==23 and training['steps']==identity['steps']==100000 and not training['resumed']
    assert training['mechanism_supervision'] and training['auxiliary_loss_weight']==1
    assert training['calibration_sha256']==identity['calibration_sha256']==expected
    fitted=json.loads((D/'fitted.json').read_text());assert fitted['calibration_sha256']==expected
    assert fitted['seed']==6801 and fitted['penalty']==1 and fitted['threshold']==0 and fitted['measurement_window']=='post-warmup'
    development=json.loads((R/'reports/evidence/measurement-window/eval-22/fitted.json').read_text());assert fitted['checkpoint_sha256']!=development['checkpoint_sha256']
    assert all(json.loads((D/'audit.json').read_text()).values())
    summary=json.loads((D/'summary.json').read_text());assert summary['checkpoint_unchanged']
    rows=list(map(json.loads,(D/'test.jsonl').read_text().splitlines()));pred=json.loads((D/'predictions.json').read_text());target=pred['target'];assert len(rows)==len(target)==80
    for i,row in enumerate(rows):
        assert row['episode']==i and row['seed']==6801 and row['measurement_window']=='post-warmup' and row['seconds']<=35
        assert target[i]==FAULT_MASK[row['fault']] and len(row['frames'])==4
        for j,frame in enumerate(row['frames']):
            assert frame['successes']==frame['requests'] and len(frame['vector'])==10
            assert set(frame['metrics'])=={'checkout','payment','ledger'}
            assert all(m['count']==(j+1)*frame['requests'] for m in frame['metrics'].values())
    train=np.load(D/'train-features.npz');test=np.load(D/'test-features.npz');probes=np.load(D/'probes.npz')
    calibration=list(map(json.loads,(R/'reports/evidence/complete-coverage/calibration.jsonl').read_text().splitlines()))
    assert len(calibration)==len(train['latent'])==160
    np.testing.assert_array_equal(train['labels'],labels(calibration))
    np.testing.assert_allclose(train['raw'],np.asarray([observations(r)[-1][2:10] for r in calibration]))
    np.testing.assert_array_equal(test['labels'],labels(rows))
    np.testing.assert_allclose(test['raw'],np.asarray([observations(r)[-1][2:10] for r in rows]))
    for name in ('raw','latent','shuffled_latent'):
        key='raw' if name=='raw' else 'latent';y=train['labels']
        if name=='shuffled_latent':y=y[np.random.default_rng(7001).permutation(len(y))]
        refit=fit_probe(train[key],y);saved={k:probes[name+'_'+k] for k in refit}
        for k in refit:np.testing.assert_allclose(refit[k],saved[k],rtol=1e-9,atol=1e-9)
        np.testing.assert_array_equal(predict(saved,test[key]),pred[name])
    methods=summary['methods']
    for name,m in methods.items():
        assert len(pred[name])==80 and m['correct']==sum(a==b for a,b in zip(pred[name],target))
        for f,v in m['families'].items():
            indices=[i for i,r in enumerate(rows) if r['fault']==int(f)]
            assert len(indices)==v['n']==10 and v['correct']==sum(pred[name][i]==target[i] for i in indices)
        assert m['gate_passed']==all(v['correct']>=(10 if f=='0' else 9) for f,v in m['families'].items())
    primary=methods['latent'];completion=json.loads((E/'completion.json').read_text());assert completion['primary']=='latent' and completion['confirmation_passed']==primary['gate_passed']
    errors=[dict(episode=i,fault=rows[i]['fault'],target=target[i],prediction=pred['latent'][i],last_metrics=rows[i]['frames'][-1]['metrics']) for i in range(80) if pred['latent'][i]!=target[i]]
    verification=dict(independent_model_seed=23,fresh_scenario_seed=6801,incidents=80,training_calibration_unchanged=True,probes_reproduced_from_calibration_only=True,predictions_reproduced=True,all_measurement_windows_valid=True,label_audits_passed=True,checkpoint_unchanged=True,protocol_unchanged=True)
    (E/'verified.json').write_text(json.dumps(verification,indent=2)+'\n');(E/'primary-errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    lines=['# Independent learned-state classifier confirmation','',f"**Primary result: {primary['correct']}/80; gate {'passed' if primary['gate_passed'] else 'failed'}.** Training seed 23, fresh scenario seed 6801. Training took {training['seconds']/60:.2f} minutes for 100,000 steps. The previous seed-22 80/80 result remains development evidence and is not pooled with this confirmation.",'',
      'This tests a frozen classifier recipe on an independently trained Dreamer representation. Coefficients are fitted on the original calibration only because a new model has different latent coordinates. No weights, thresholds or normalization were fitted on confirmation incidents.','',
      '| Readout | Exact diagnoses | Per-family gate |','|---|---:|---|']
    for name,m in methods.items():lines.append(f"| {name}{' (primary)' if name=='latent' else ''} | {m['correct']}/80 | {'pass' if m['gate_passed'] else 'fail'} |")
    lines += ['', '| Family | Latent primary | Raw telemetry | Reward |','|---|---:|---:|---:|']
    for f,name in enumerate(FAMILIES):lines.append('| '+name+' | '+' | '.join(str(methods[k]['families'][str(f)]['correct'])+'/10' for k in ('latent','raw','dreamer_reward'))+' |')
    lines += ['', '## Interpretation','',
      ('The prespecified independent confirmation passed. This supports a small known-regime triage POC using a classifier over a Dreamer-trained state.' if primary['gate_passed'] else 'The prespecified independent confirmation failed. The promising 80/80 development result did not establish a reproducible acceptance pass.'),
      'The candidate uses Dreamer to learn the recurrent system representation and a supervised ridge classifier to select the fault set. It does not use imagined reward to make the final diagnosis. Fixed inspections mean this experiment does not demonstrate learned investigation planning or autonomous remediation.',
      'The raw-telemetry baseline and shuffled-label control remain necessary comparisons. One small confirmation batch cannot prove world-model superiority, production reliability or transfer to unseen services. All eight fault regimes were represented in training. Historical calibration includes warm-up measurements, while fresh evaluation uses clean steady-state windows. Startup handling remains untested.','',
      '## Next step','',
      ('Add the frozen classifier to the inference-only API and verify action parity with this evaluation before any namespace-scoped serving promotion. Then evaluate fresh live incidents through that API. LLM orchestration and recovery actions remain separate work.' if primary['gate_passed'] else 'Inspect the preserved confirmation errors and compare their telemetry with the calibration distribution. Any revised candidate needs a new prespecified test; do not tune on this batch and call it confirmation.'),'',
      '## Validation and isolation','',
      'Three existing classifier tests and four SSH-monitor regression tests passed. Verification reproduced probe coefficients from saved calibration features and every probe prediction from saved test features, checked all runtime label-isolation audits, confirmed checkpoint immutability and validated all 80 measurement windows.',
      'Original service source bundles and all five baseline deployment replicas were restored. No candidate was promoted and no other namespace was modified. The generic evaluator summary describes an individual batch; confirmation status is defined by this preregistered protocol and completion record.','',
      'Evidence: `reports/evidence/latent-confirmation/`. Protocol: `reports/LATENT_CONFIRMATION_PROTOCOL.md`.']
    (R/'reports/LATENT_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'primary':primary,'verified':verification},indent=2))

if __name__=='__main__':main()
