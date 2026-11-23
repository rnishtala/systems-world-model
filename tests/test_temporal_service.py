import importlib.util,os
from pathlib import Path
from unittest.mock import patch
import pytest

@pytest.fixture
def service():
    spec=importlib.util.spec_from_file_location('temporal_service',Path(__file__).parents[1]/'microservice/service.py');module=importlib.util.module_from_spec(spec)
    with patch.dict(os.environ,{'SERVICE_ROLE':'ledger'}):spec.loader.exec_module(module)
    return module

def test_retry_is_opt_in_and_bounded(service):
    calls=[];counts=[]
    def timeout(url,budget):calls.append(budget);raise TimeoutError()
    with pytest.raises(TimeoutError):service.downstream_call('test',service.DEFAULT,timeout,record=counts.append)
    assert len(calls)==1
    calls.clear();counts.clear()
    cfg=dict(service.DEFAULT,retry_limit=2,attempt_timeout=.1,request_budget=1)
    with pytest.raises(TimeoutError):service.downstream_call('test',cfg,timeout,record=counts.append)
    assert len(calls)==3 and counts.count('retries')==2 and counts.count('downstream_timeouts')==3

def test_total_retry_budget_prevents_another_attempt(service):
    now=[0];calls=[]
    def timeout(url,budget):calls.append(budget);now[0]+=.1;raise TimeoutError()
    cfg=dict(service.DEFAULT,retry_limit=2,attempt_timeout=.1,request_budget=.15,retry_backoff=.06)
    with pytest.raises(TimeoutError):service.downstream_call('test',cfg,timeout,clock=lambda:now[0],record=lambda _:None)
    assert len(calls)==1

def test_controls_preserve_continuous_windows_and_same_pool(service):
    service.apply({});pool=service.POOL
    row=dict(sequence=1,elapsed=.2,cpu=0,pool_wait=0,db=0,downstream=0,self_time=.2,error=0)
    service.ROWS.append(row);service.SEQUENCE=1
    service.apply({'db_hold':.12})
    assert service.POOL is pool and service.snapshot(0)['count']==1
    assert service.snapshot(1)['count']==0
    service.reset_metrics();assert service.snapshot(0)['window_gap']

def test_pool_retirement_closes_idle_connections(service):
    class Conn:
        closed=False
        def close(self):self.closed=True
    service.apply({});old=service.POOL;conn=Conn();old.idle.put(conn)
    service.apply({'pool_size':2})
    assert old.retired and conn.closed

@pytest.mark.parametrize('cfg',[{'retry_limit':3},{'retry_limit':True},{'attempt_timeout':0},{'request_budget':9},{'retry_backoff':.2}])
def test_invalid_retry_controls_rejected(service,cfg):
    with pytest.raises(ValueError):service.apply(cfg)


def test_real_http_timeouts_retry_without_hiding_completed_request(service):
    import http.server,threading,time,json,urllib.request,urllib.error
    class Slow(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            time.sleep(.15)
            try:
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'trace':[]}).encode())
            except (BrokenPipeError,ConnectionResetError):pass
    upstream=http.server.ThreadingHTTPServer(('127.0.0.1',0),Slow)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),service.Handler)
    service.ROLE='payment';service.NEXT='http://127.0.0.1:'+str(upstream.server_port)
    service.apply({'retry_limit':1,'attempt_timeout':.05,'request_budget':.2})
    threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in (upstream,server)]
    for t in threads:t.start()
    try:
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port)+'/request',timeout=2)
        counters=service.snapshot()['counters']
        assert counters['downstream_attempts']==2 and counters['downstream_timeouts']==2 and counters['retries']==1
        assert counters['original_requests']==1 and counters['completed_requests']==1 and counters['active_requests']==0
        assert service.snapshot(0)['count']==1
    finally:
        for s in (server,upstream):s.shutdown();s.server_close()
