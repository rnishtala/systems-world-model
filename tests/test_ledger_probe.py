import copy
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from scripts.evaluate_ledger_probe import features

def test_features_ignore_truth_and_probe_order():
    metrics={s:dict(latency_p95=.2,self_p95=.04,downstream_p95=.1,error_rate=0,count=3) for s in ('checkout','payment','ledger')}
    a=dict(requests=3,metrics=metrics);b=dict(requests=9,metrics=metrics)
    row=dict(base=[a,b,a,b],extra=dict(checkout=[a,b],ledger=[a,b]))
    x=features([row]);assert x['ledger'].shape==(1,17)
    np.testing.assert_array_equal(x['checkout'],x['ledger'])
    changed=copy.deepcopy(row);changed.update(fault=7,seed=999,repeat=42)
    changed['extra']['ledger'].reverse()
    np.testing.assert_array_equal(x['ledger'],features([changed])['ledger'])

def test_measurement_contract(monkeypatch):
    live=types.ModuleType('live');live.SERVICES=('checkout','payment','ledger')
    monkeypatch.setitem(sys.modules,'live',live)
    path=Path(__file__).resolve().parents[1]/'microservice/ledger_probe.py'
    spec=importlib.util.spec_from_file_location('probe_test',path)
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    counts={};resets=[]
    def reset():counts.clear();resets.append(True)
    def request(service,path):
        return dict(count=counts.get(service,0),latency_p95=.2,self_p95=.1,downstream_p95=.1,error_rate=0,cpu_mean=999,pool_wait_p95=999,db_mean=999)
    def batch(target,n):
        for s in live.SERVICES:counts[s]=n if target=='checkout' or s=='ledger' else 0
        return n
    live.request=request;live.reset_measurement_window=reset
    monkeypatch.setattr(probe,'batch',batch)
    for target in ('checkout','ledger'):
        out=probe.frame(target,3);assert out['successes']==3
        for m in out['metrics'].values():assert not {'pool_wait_p95','cpu_mean','db_mean'} & set(m)
    assert len(resets)==2
    monkeypatch.setattr(probe,'batch',lambda target,n:0)
    with pytest.raises(RuntimeError):probe.frame('ledger',3)


def test_monitor_timeout_does_not_delete_or_restart_job(monkeypatch,tmp_path):
    import json
    from scripts import run_ledger_probe as runner
    out=tmp_path/'evidence';monkeypatch.setattr(runner,'O',out)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:None)
    events=[];state={'failed_once':False,'split':'development'}
    def run(cmd,body=None):
        events.append(cmd)
        if cmd=='get jobs -o json':return json.dumps({'items':[]})
        if cmd=='get deployment swm-inference -o json':return json.dumps({'spec':{'replicas':1,'template':{}}})
        if cmd=='get resourcequota lab-budget -o json':return json.dumps({'status':{'used':{'limits.cpu':'1'}}})
        if cmd=='create -f -':
            item=json.loads(body)
            if item['kind']=='Job':state['split']=item['spec']['template']['spec']['containers'][0]['command'][-1]
        if cmd.startswith('get job '):
            if not state['failed_once']:
                state['failed_once']=True;events.append('TIMEOUT');raise RuntimeError('SSH timeout')
            events.append('TERMINAL_SUCCESS');return json.dumps({'status':{'succeeded':1}})
        if cmd.startswith('logs '):
            return '\n'.join(['ROW {}']*32+['RESET_COMPLETE','COMPLETE '+state['split']])
        return ''
    monkeypatch.setattr(runner,'run',run);runner.main()
    between=events[events.index('TIMEOUT')+1:events.index('TERMINAL_SUCCESS')]
    assert not any(x.startswith(('delete','create','scale')) for x in between)
    assert (out/'monitor-errors.jsonl').exists()
    assert len((out/'test.jsonl').read_text().splitlines())==32
    assert json.loads((out/'restored.json').read_text())['replicas']==1
