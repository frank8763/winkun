"""Paired on-policy pilot: bounded endpoint proximity cost versus original MC."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t267_causal_credit_course'))
from t267_learning import *
import math


def endpoint_cost(arrays,cap):
    # An SMDP endpoint reward, not an exact integral over unrecorded native XY.
    # Current goal range is observation column48 (normalized by10m).
    n=arrays['native_counts'].astype(np.int64)
    fraction=np.clip(arrays['next_states'][:,48].astype(np.float64)/max(float(arrays['states'][0,48]),.02),0,1)
    return (1-R.GAMMA**n)/(1-R.GAMMA**cap)*fraction


def discounted_cost(cost,counts):
    return float(np.dot(R.GAMMA**np.r_[0,np.cumsum(counts.astype(np.int64))[:-1]],cost))


def reward_screen(items):
    results=[];informative=False
    for name in sorted({i['row']['source_reset_id'] for i in items}):
        group=[i for i in items if i['row']['source_reset_id']==name]
        g=[float(S.monte_carlo(i['arrays']['rewards'],i['arrays']['native_counts'])[0]) for i in group]
        original=[float(S.monte_carlo(i['arrays']['environment_rewards'],i['arrays']['native_counts'])[0]) for i in group]
        costs=[discounted_cost(i['arrays']['aux_costs'],i['arrays']['native_counts']) for i in group]
        timeout=[k for k,i in enumerate(group) if i['row']['timeout'] and i['row']['complete'] and not i['row']['collision']]
        spreads=None
        if len(timeout)>=2:
            a=float(np.ptp(np.array(original)[timeout]));b=float(np.ptp(np.array(g)[timeout]))
            spreads=dict(same_cap_timeouts=len(timeout),original_return_spread=a,new_return_spread=b)
            informative|=a<1e-10 and b>1e-5
        assert all(-1e-12<=c<=1+1e-12 for c in costs)
        results.append(dict(reset=name,G0=g,original_G0=original,unit_cost=costs,timeout_comparison=spreads))
    return dict(passed=informative,groups=results,criterion='At least2 actual equal-cap timeouts in one fixed reset; original G0 spread<1e-10 and bounded-cost G0 spread>1e-5. No replacement if absent.')
