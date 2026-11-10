"""Audit live diagnosis and forecasts against a calibration-only baseline."""
import argparse,collections,json,statistics,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--results',required=True);p.add_argument('--confirmation',action='store_true');a=p.parse_args()
cal=[json.loads(x) for x in Path(a.calibration).read_text().splitlines()];rows=[json.loads(x) for x in Path(a.results).read_text().splitlines()]
gain=[10,1,1,3,3,10,4,10,10,10] if rows and rows[0].get('observation_version')=='physical-v2' else [1]*10
assert len({r.get('observation_version','original') for r in rows})==1
means={i:statistics.mean(min(2,max(0,f['vector'][i]*gain[i])) for r in cal for f in r['frames']) for i in range(10)}
def wilson(correct,n):
 z=1.95996398454;p=correct/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,mid-half),min(1,mid+half)]
out={'physical_incidents':len({r['episode'] for r in rows}),'calibration_incidents':len(cal),'methods':{}}
for method in sorted({str(r['method']) for r in rows}):
 rs=[r for r in rows if str(r['method'])==method];classes={}
 for fault in sorted({r['fault'] for r in rs}):
  c=[r for r in rs if r['fault']==fault]
  assert len({r['episode'] for r in c})==len(c)
  assert all(r['correct']==(r['diagnosis']==r['fault']) for r in c)
  classes[str(fault)]={'n':len(c),'accuracy':statistics.mean(r['correct'] for r in c),'accuracy_wilson95':wilson(sum(r['correct'] for r in c),len(c)),'false_fault_rate':statistics.mean(r['diagnosis']>0 for r in c),'unresolved':statistics.mean(r['diagnosis']==-1 for r in c)}
 errors=[];base=[];per=collections.defaultdict(lambda:[[],[]]);count=0
 for r in rs:
  for forecast in r['predictions']:
   count+=1
   for i,pred,actual in zip(forecast['indices'],forecast['predicted'],forecast['actual']):
    e=(pred-actual)**2;b=(means[i]-actual)**2;errors.append(e);base.append(b);per[i][0].append(e);per[i][1].append(b)
 m={'evaluations':len(rs),'classes':classes,'accuracy':statistics.mean(r['correct'] for r in rs),'checks':statistics.mean(r['checks'] for r in rs),'median_seconds':statistics.median(r['seconds'] for r in rs),'forecast_count':count,'prediction_mse':statistics.mean(errors) if errors else None,'baseline_mse':statistics.mean(base) if base else None,'per_sensor':{str(i):{'model_mse':statistics.mean(x[0]),'baseline_mse':statistics.mean(x[1])} for i,x in per.items()}}
 m['triage_thresholds_met']=all(str(f) in classes and classes[str(f)]['accuracy']>=.9 for f in (1,2,3)) and '0' in classes and classes['0']['false_fault_rate']<=.05 and classes['0']['accuracy']>=.95 and m['checks']<=3 and m['median_seconds']<60
 m['prediction_thresholds_met']=bool(count>=30 and m['prediction_mse']<m['baseline_mse'])
 out['methods'][method]=m
out['frozen_confirmation_passed']=bool(a.confirmation and all(str(s) in out['methods'] and out['methods'][str(s)]['triage_thresholds_met'] and out['methods'][str(s)]['prediction_thresholds_met'] and all(out['methods'][str(s)]['classes'].get(str(f),{}).get('n')==30 for f in range(4)) for s in (10,11,12)))
print(json.dumps(out,indent=2))
