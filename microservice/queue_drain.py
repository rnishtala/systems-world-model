"""Paired screening experiment for recovery dynamics after load reduction."""
import argparse
import json
import os
from pathlib import Path
import random
from statistics import median

from microservice.temporal_pilot import incident

HOLDS_MS = (90, 93, 95, 97, 99, 101)
ORDERS = ('building', 'draining')


def trajectory(result):
    anchor = result['action_boundary']
    post = [w for w in result['windows'] if w['elapsed'] > anchor['elapsed']]
    previous = anchor['counters']
    fractions = []
    counts = []
    for end in (2, 4, 6, 8):
        boundary = next(w for w in post if w['elapsed'] >= anchor['elapsed'] + end)
        if boundary['elapsed'] - anchor['elapsed'] - end > .8:
            raise ValueError('Sampling gap')
        counters = boundary['metrics']['checkout']['counters']
        n, failed = (counters.get(k, 0) - previous.get(k, 0)
                     for k in ('completed_requests', 'failed_requests'))
        if n < 2 or not 0 <= failed <= n:
            raise ValueError('Incomplete outcome bin')
        fractions.append(failed / n)
        counts.append(n)
        previous = counters
    # Report the nominal end of the second consecutive clean bin.
    recovery = next((2*(i+1) for i in range(1,4) if fractions[i-1:i+1] == [0,0]), None)
    prefix = [w for w in result['windows'] if w['elapsed'] < 6]
    return dict(fractions=fractions, counts=counts, recovery=recovery,
                waiters=median(w['metrics']['ledger']['counters'].get('pool_waiters', 0)
                               for w in prefix[-2:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    if os.environ.get('POD_NAMESPACE') != 'systems-world-model-lab':
        raise ValueError('Wrong namespace')
    cases = [(ms,repeat,order) for ms in HOLDS_MS for repeat in range(2) for order in ORDERS]
    random.Random(args.seed).shuffle(cases)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(cases=cases, seed=args.seed, horizon_seconds=8, minimum_completions_per_bin=2,
                    recovery='End of two consecutive two-second bins with zero failed requests',
                    censoring='No confirmed recovery by horizon; retained as null',
                    gate='At least two service times where both repeats show greater pre-action waiters and later confirmed recovery for building; censored versus recovered counts, two censored do not.')
    (args.out/'protocol.json').write_text(json.dumps(protocol))
    rows = {}
    for i,(ms,repeat,order) in enumerate(cases):
        profile = dict(fault_hold=ms/1000, healthy_hold=.04, onset=1, relief_at=6,
                       duration=16, base_rate=8, burst_order=order)
        for attempt in range(3):
            result = incident('database','load',profile)
            if not result['restored']:
                raise ValueError('Lab restoration failed')
            try:
                stats = trajectory(result)
            except (StopIteration, ValueError):
                (args.out/f'incomplete-{i}-{attempt}.json').write_text(json.dumps(result))
                if attempt == 2:
                    raise
                continue
            break
        rows[ms,repeat,order] = stats
        (args.out/f'incident-{i}.json').write_text(json.dumps(dict(case=cases[i], **result, stats=stats)))
    pairs = []
    for ms in HOLDS_MS:
        for repeat in range(2):
            b,d = (rows[ms,repeat,o] for o in ORDERS)
            later = d['recovery'] is not None and (b['recovery'] is None or b['recovery'] > d['recovery'])
            pairs.append(dict(ms=ms,repeat=repeat,building=b,draining=d,
                              supports=b['waiters'] > d['waiters'] and later))
    qualifying = [ms for ms in HOLDS_MS if all(p['supports'] for p in pairs if p['ms']==ms)]
    summary = dict(queue_drain=dict(pairs=pairs, qualifying_ms=qualifying, passes=len(qualifying)>=2),incidents=[])
    (args.out/'summary.json').write_text(json.dumps(summary))


if __name__ == '__main__':
    main()
