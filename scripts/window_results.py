"""Verify and summarize the fixed-window experiment without changing its evidence."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/measurement-window'
FAMILIES=['Healthy','CPU','Pool','Delay','CPU + delay','CPU + pool','Pool + delay','Triple']

def main():
    assert json.loads((E/'restored.json').read_text())['all_deployments_ready']
    completion=json.loads((E/'completion.json').read_text());seeds=[22,23] if completion['confirmation_passed'] is not None else [22]
    lines=['# Post-warm-up measurement results','',
      'The measurement-only reset excludes completed warm-up requests while preserving fault configuration, lease expiry and pooled database connections. The previous failed run is preserved unchanged.','',
      'The trained model and calibration were fixed. The historical calibration still includes warm-up samples: this experiment isolates a measurement change, not a retraining improvement.','']
    verified={};all_errors={}
    for seed in seeds:
        D=E/f'eval-{seed}';summary=json.loads((D/'summary.json').read_text());pred=json.loads((D/'predictions.json').read_text());rows=list(map(json.loads,(D/'test.jsonl').read_text().splitlines()));fitted=json.loads((D/'fitted.json').read_text())
        assert all(json.loads((D/'audit.json').read_text()).values())
        assert len(rows)==80 and summary['checkpoint_unchanged']
        assert fitted['measurement_window']=='post-warmup'
        assert fitted['calibration_sha256']==hashlib.sha256((R/'reports/evidence/complete-coverage/calibration.jsonl').read_bytes()).hexdigest()
        if seed==22:
            old=json.loads((R/'reports/evidence/complete-coverage/eval-22/fitted.json').read_text())
            assert fitted['checkpoint_sha256']==old['checkpoint_sha256']
        for row in rows:
            assert row['measurement_window']=='post-warmup' and row['seconds']<=35
            for j,frame in enumerate(row['frames']):
                assert frame['successes']==frame['requests']
                assert all(m['count']==(j+1)*frame['requests'] for m in frame['metrics'].values())
        methods=summary['methods'];target=pred['target'];errors=[]
        for name,result in methods.items():
            assert len(pred[name])==len(target)==80
            for f,v in result['families'].items():
                selected=[i for i,r in enumerate(rows) if r['fault']==int(f)]
                assert len(selected)==v['n']==10
                assert v['correct']==sum(pred[name][i]==target[i] for i in selected)
            assert result['correct']==sum(a==b for a,b in zip(pred[name],target))
            assert result['gate_passed']==all(v['correct']>=(10 if f=='0' else 9) for f,v in result['families'].items())
        for i,row in enumerate(rows):
            wrong={name:pred[name][i] for name in methods if name!='shuffled_latent' and pred[name][i]!=target[i]}
            if wrong:errors.append({'episode':i,'fault':row['fault'],'target':target[i],'wrong':wrong,'last_metrics':row['frames'][-1]['metrics']})
        all_errors[str(seed)]=errors;verified[str(seed)]={'incidents':80,'measurement_counts_valid':True,'checkpoint_unchanged':True,'scenario_seed':fitted['seed']}
        lines += [f'## Model seed {seed}','', '| Readout | Exact diagnoses | Gate |','|---|---:|---|']
        for name,m in methods.items():lines.append(f"| {name} | {m['correct']}/80 | {'pass' if m['gate_passed'] else 'fail'} |")
        lines += ['', '| Family | Dreamer reward | Auxiliary | Telemetry |','|---|---:|---:|---:|']
        for f,name in enumerate(FAMILIES):lines.append('| '+name+' | '+' | '.join(str(methods[m]['families'][str(f)]['correct'])+'/10' for m in ('dreamer_reward','dreamer_auxiliary','raw'))+' |')
        lines += ['']
    lines += ['## Decision and limits','',f"Development gate: {'passed' if completion['development_passed'] else 'failed'}. Independent confirmation: "+('skipped after development failure.' if completion['confirmation_skipped'] else ('passed.' if completion['confirmation_passed'] else 'failed.')),
      'No candidate was promoted. All original service sources and five deployment replicas were restored. No other namespace was modified.','',
      'This is a small experiment across eight known fault regimes, not a production reliability estimate. Startup transients were intentionally excluded and need their own scenario. The telemetry-only comparator is essential: a high Dreamer score alone does not demonstrate an advantage from its learned world model. LLM-assisted resolution and unseen-system generalization remain untested.','',
      'All three primary errors omitted pool contention: two CPU + pool incidents and one triple. The frozen latent-state classifier scored 80/80, while the shuffled-label control scored 8/80. This suggests the learned representation contains diagnostic information that the reward readout does not consistently use. It remains a secondary finding from one small batch, not an independent validation or a retroactive primary success. Next: prespecify this fixed latent readout and test it with an independent model and fresh incidents.', '',
      'Validation: four measurement/control-boundary tests and four SSH-monitor regression tests passed. Runtime label-isolation audits, checkpoint immutability and all 80 incident window counts are preserved with the results.','',
      'Evidence: `reports/evidence/measurement-window/`. Protocol: `reports/MEASUREMENT_WINDOW_PROTOCOL.md`.']
    (E/'verified.json').write_text(json.dumps(verified,indent=2)+'\n');(E/'error-analysis.json').write_text(json.dumps(all_errors,indent=2)+'\n');(R/'reports/MEASUREMENT_WINDOW_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
