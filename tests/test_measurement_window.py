"""Measurement resets must not reset the physical incident or expose controls."""
import importlib.util,io,json,os,unittest
from pathlib import Path
from unittest.mock import patch

class MeasurementWindowTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('window_service',Path(__file__).parents[1]/'microservice/service.py')
        self.service=importlib.util.module_from_spec(spec)
        with patch.dict(os.environ,{'SERVICE_ROLE':'ledger'}):spec.loader.exec_module(self.service)
        self.service.apply({'pool_size':1,'delay_seconds':.3,'lease_seconds':45})
        self.service.ROWS.append(dict(elapsed=9,cpu=0,pool_wait=0,db=0,downstream=0,self_time=9,error=0))
    def post(self,control,path,body):
        handler=object.__new__(self.service.Handler);handler.control=control;handler.path=path
        encoded=json.dumps(body).encode();handler.headers={'Content-Length':str(len(encoded))};handler.rfile=io.BytesIO(encoded)
        answers=[];handler.reply=lambda data,status=200:answers.append((status,data));handler.do_POST();return answers[0][0]
    def test_reset_preserves_pool_connections_config_and_lease(self):
        s=self.service;pool=s.POOL;connection=object();pool.idle.put(connection);config=s.CONFIG;expires=s.EXPIRES
        self.assertEqual(self.post(True,'/metrics/reset',{}),200)
        self.assertEqual(s.snapshot()['count'],0);self.assertIs(s.POOL,pool);self.assertIs(pool.idle.get_nowait(),connection)
        self.assertIs(s.CONFIG,config);self.assertEqual(s.EXPIRES,expires)
        s.ROWS.append(dict(elapsed=.05,cpu=0,pool_wait=0,db=0,downstream=0,self_time=.05,error=0))
        self.assertEqual(s.snapshot()['latency_p95'],.05)
    def test_data_port_cannot_reset_metrics(self):
        self.assertEqual(self.post(False,'/metrics/reset',{}),404);self.assertEqual(len(self.service.ROWS),1)
    def test_reset_rejects_nonempty_or_nonobject_body(self):
        for body in ({'pool_size':8},[],None):
            self.assertEqual(self.post(True,'/metrics/reset',body),400);self.assertEqual(len(self.service.ROWS),1)
    def test_operator_uses_only_measurement_control_endpoint(self):
        from microservice import live
        with patch.object(live,'request') as call:live.reset_measurement_window()
        self.assertEqual(call.call_args_list,[unittest.mock.call(s,'/metrics/reset',{},True) for s in live.SERVICES])

if __name__=='__main__':unittest.main()
