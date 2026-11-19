"""Synthetic contract fixture, not experimental evidence."""
def incident(seed=1,fault=0,repeat=0,regime=0):
    def frame(target,n):
        metrics={s:dict(count=n if target=='checkout' or s=='ledger' else 0,latency_p95=.1,self_p95=.02,downstream_p95=.08,error_rate=0.) for s in ('checkout','payment','ledger')}
        return dict(target=target,requests=n,successes=n,metrics=metrics)
    return dict(seed=seed,fault=fault,repeat=repeat,regime=regime,utc=1.,base=[frame('checkout',n) for n in (3,9,3,9)],extra={t:[frame(t,n) for n in (3,9)] for t in ('checkout','ledger')},order=['checkout','ledger'])
