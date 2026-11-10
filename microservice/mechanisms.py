"""Versioned mechanism labels, kept outside the observation vector.
Fault IDs describe operator interventions; diagnosis masks use CPU=1,pool=2,delay=4.
"""
CONTRACT = 'mechanisms-v1'
FAULT_MASK = {0:0, 1:1, 2:2, 3:4, 4:5, 5:3, 6:6, 7:7}
NAMES = ('cpu', 'pool', 'delay')
def diagnosis_name(mask):
    if mask == -1:return 'escalation'
    if mask not in range(8):raise ValueError(mask)
    return '+'.join(name for i,name in enumerate(NAMES) if mask & (1 << i)) or 'healthy'

def describe_action(action,contract):
    if contract not in ('single-v1',CONTRACT):raise ValueError('Unknown action contract')
    end=12 if contract==CONTRACT else 8
    if not 0<=action<=end:raise ValueError('Invalid action')
    if action<4:return False,('wait','inspect_cpu','inspect_pool','inspect_trace')[action]
    if action==end:return True,'escalation'
    if contract==CONTRACT:return True,diagnosis_name(action-4)
    return True,('healthy','payment:cpu_contention','ledger:pool_contention','ledger:downstream_delay')[action-4]
