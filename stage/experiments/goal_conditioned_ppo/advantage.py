"""Finite-task semi-Markov advantage contract, not a trained algorithm.

gamma is per native physics step; lambda is per nominal 100-step decision.
Task timeouts are genuine finite-horizon terminals. Administrative censoring
is refused here rather than silently converted into a terminal training row.
"""
import numpy as np

def advantage(rewards, native_counts, values, end_flags, *, gamma, lam=0.95, complete=True):
    rewards=np.asarray(rewards,dtype=np.float64)
    counts=np.asarray(native_counts,dtype=np.int64)
    values=np.asarray(values,dtype=np.float64)
    flags=np.asarray(end_flags,dtype=bool)
    n=len(rewards)
    if not complete:raise ValueError('Administrative censoring requires a separate bootstrap protocol')
    if n==0 or counts.shape!=(n,) or values.shape!=(n+1,) or flags.shape!=(n,3):raise ValueError('Invalid trajectory shapes')
    if not (0<gamma<=1 and 0<=lam<=1) or not np.all((counts>0)&(counts<=100)):raise ValueError('Invalid native duration or discount')
    if flags[:-1].any() or not flags[-1].any():raise ValueError('Only one complete finite-task trajectory is supported')
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():raise ValueError('Nonfinite values')
    discounts=np.power(gamma,counts)
    continuation=(~flags.any(axis=1)).astype(np.float64)
    # All CONTACT, GOAL and true finite-task cap terminals have bootstrap zero.
    delta=rewards+discounts*continuation*values[1:]-values[:-1]
    trace=np.power(lam,counts/100.)
    result=np.empty(n,dtype=np.float64);carry=0.
    for k in range(n-1,-1,-1):
        carry=delta[k]+discounts[k]*trace[k]*continuation[k]*carry
        result[k]=carry
    return dict(advantage=result,value_target=result+values[:-1],delta=delta,
                discount=discounts,lambda_discount=trace)
