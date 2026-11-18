"""Uncapped trace view and fixed train/validation/test workload regimes."""
import numpy as np
from microservice.temporal import TemporalEnv, TemporalIncident
PROFILES={'train':[(2,4,2,4),(4,6,4,6),(6,8,6,8),(8,10,8,10)],
          'validation':[(3,7,3,7),(5,9,5,9)],'test':[(3,11,6,9)]}
REPEATS={'train':4,'validation':4,'test':10}
SEEDS={'train':8701,'validation':8702,'test':8703}

def cases(split):
    return [(f,r,j%2) for f in range(8) for r in range(len(PROFILES[split])) for j in range(REPEATS[split])]

def loads(split,regime,phase):
    values=list(PROFILES[split][regime])
    return values[::-1] if phase else values

def vector(row,t):
    frame=row['frames'][min(t,3)];m=frame['metrics']
    latency=np.array([m['checkout']['latency_p95'],m['payment']['self_p95'],
                      m['ledger']['latency_p95'],m['payment']['downstream_p95']],float)
    if not np.isfinite(latency).all() or np.any(latency<0):raise ValueError('Invalid observed latency')
    error=float(m['checkout']['error_rate']);load=float(frame['requests'])/12
    if not 0<=error<=1 or not 0<load<=1:raise ValueError('Invalid observed error/load')
    x=np.log1p(latency/.1)
    return np.array([x[0],error,load,x[1],x[2],x[3],0,0,0,0,
                     *([float(t>=3)]*3),min(t,7)/7],np.float32)

class LogIncident(TemporalIncident):
    def observe(self):return vector({'frames':self.frames},self.t)

class LogEnv(TemporalEnv):
    incident_type=LogIncident
