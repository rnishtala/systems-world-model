"""Load-response features from observed traces only."""
import numpy as np
from microservice.regime import vector

def response(row):
    # Observed request counts and trace latencies only; no labels/controls/counters.
    x=np.array([vector(row,t)[[0,3,4,5]] for t in range(4)])
    n=np.array([f['requests'] for f in row['frames']],float)
    low=x[n==n.min()].mean(0);high=x[n==n.max()].mean(0)
    slope=(high-low)/(n.max()-n.min())
    return np.concatenate([low,high,slope,[n.min()/12,n.max()/12]])

