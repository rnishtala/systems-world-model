"""Empirical forecast-error guard; diagnostic reward scores are not confidence."""
import math
GROUPS={1:(3,4),2:(5,6),3:(7,8,9)}

def residuals(predictions):
    values={}
    for item in predictions:
        indices=item['indices'];predicted=item['predicted'];actual=item['actual']
        if len(indices)!=len(predicted) or len(indices)!=len(actual):raise ValueError('Malformed forecast')
        for i,p,a in zip(indices,predicted,actual):
            if i in values:raise ValueError('Duplicate inspected sensor')
            if not math.isfinite(p) or not math.isfinite(a):raise ValueError('Nonfinite forecast')
            values[i]=abs(a-p)
    if set(values)!=set(range(3,10)):raise ValueError('All three inspections required')
    return values

def assess(config,seed,diagnosis,predictions):
    errors=residuals(predictions)
    limits=config['seeds'][str(seed)]['classes'][str(diagnosis)]['limits']
    exceeded=[{'index':i,'absolute_error':errors[i],'limit':limits[str(i)]} for i in sorted(errors) if errors[i]>limits[str(i)]]
    return {'escalate':bool(exceeded),'reason':'forecast_contradiction' if exceeded else 'within_calibrated_bounds','exceeded':exceeded,'version':config['version'],'scope':'Empirical bounds; not calibrated confidence or a general unknown-fault guarantee.'}
