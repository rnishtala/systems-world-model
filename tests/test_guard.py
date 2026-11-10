import pytest
from microservice.guard import assess,residuals

def history(value=0):
 return [{'indices':list(range(3,10)),'predicted':[0.]*7,'actual':[value]*7}]

def config():return {'version':'test','seeds':{'10':{'classes':{'1':{'limits':{str(i):.2 for i in range(3,10)}}}}}}

def test_unexplained_observation_escalates_even_with_known_candidate():
 r=assess(config(),10,1,history(.21));assert r['escalate'] and len(r['exceeded'])==7
 assert not assess(config(),10,1,history(.2))['escalate']

def test_missing_duplicate_and_nonfinite_evidence_rejected():
 with pytest.raises(ValueError):residuals([])
 with pytest.raises(ValueError):residuals(history()+history())
 with pytest.raises(ValueError):residuals(history(float('nan')))

def test_unknown_model_has_no_silent_fallback():
 with pytest.raises(KeyError):assess(config(),11,1,history())

def test_combined_controls_preserve_pool_and_delay(monkeypatch):
 from microservice import live
 class Rng:
  def uniform(self,a,b):return (a+b)/2
  def integers(self,a,b):return a
 calls=[];monkeypatch.setattr(live,'reset',lambda:None)
 monkeypatch.setattr(live,'request',lambda service,path,body,control:calls.append((service,body)))
 live.configure(6,Rng(),True)
 assert calls[-1][0]=='ledger' and calls[-1][1]['pool_size']==1 and calls[-1][1]['delay_seconds']>0
