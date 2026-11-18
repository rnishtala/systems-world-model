"""Fixed forecast-informed diagnostic features, after four observations arrive."""
import numpy as np

def features(base,forecast):
    n=len(base['response'])
    change=(forecast['fixed']-forecast['current']).reshape(n,12)
    residual=(forecast['actual']-forecast['fixed']).reshape(n,12)
    return dict(response=base['response'],history=base['history'],latent=base['latent'],
        response_history=np.concatenate([base['response'],base['history']],1),
        response_latent=np.concatenate([base['response'],base['latent']],1),
        response_forecast=np.concatenate([base['response'],change,residual],1))
