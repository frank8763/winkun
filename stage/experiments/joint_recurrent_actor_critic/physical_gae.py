"""NumPy reference for physical-time GAE; no simulator or learning side effects.

This is an unregistered engineering component, not a runnable experiment.
Rewards must already include discounting *within* each action hold. All arrays
have shape (decision, environment). The caller must supply the actual final
observation's value, never an automatically reset observation's value.
"""

import numpy as np


def physical_gae(
    rewards,
    values,
    next_values,
    native_counts,
    bootstrap_allowed,
    trace_continues,
    *,
    gamma_native,
    lambda_native,
):
    """Return float64 advantages and value targets without modifying inputs.

    ``bootstrap_allowed`` is false at true task termination, including an
    inherent finite task deadline. It may be true at a rollout cut with a valid
    final observation. ``trace_continues`` connects adjacent rows of this same
    environment/episode and must be false at resets, missing/censored
    continuations, and the last buffer row. A truncation may bootstrap without
    connecting to a reset episode. Invalid/censored data without a legitimate
    final observation must be excluded by the caller, not invented here.

    Both gamma and lambda are per-native-step factors. Consequently the trace
    decay of a hold of n steps is (gamma_native * lambda_native)**n. This is a
    physical-time variant of standard GAE, not a claim of a novel estimator.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    vn = np.asarray(next_values, dtype=np.float64)
    counts = np.asarray(native_counts)
    bootstrap = np.asarray(bootstrap_allowed)
    trace = np.asarray(trace_continues)
    if r.ndim != 2 or min(r.shape) == 0:
        raise ValueError('Expected nonempty (decision, environment) arrays')
    if any(a.shape != r.shape for a in (v, vn, counts, bootstrap, trace)):
        raise ValueError('All arrays must have the same shape')
    if not all(np.isfinite(a).all() for a in (r, v, vn)):
        raise ValueError('Nonfinite rewards or values')
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
        raise ValueError('native_counts must contain positive integers')
    if bootstrap.dtype != np.bool_ or trace.dtype != np.bool_:
        raise ValueError('Termination and continuation masks must be boolean')
    if np.any(trace & ~bootstrap):
        raise ValueError('A terminal transition cannot continue the GAE trace')
    if trace[-1].any():
        raise ValueError('The last buffer row cannot continue an absent trace')
    if not (np.isfinite(gamma_native) and 0 < gamma_native <= 1):
        raise ValueError('gamma_native must be in (0, 1]')
    if not (np.isfinite(lambda_native) and 0 <= lambda_native <= 1):
        raise ValueError('lambda_native must be in [0, 1]')

    discounts = np.power(float(gamma_native), counts)
    decay = np.power(float(gamma_native) * float(lambda_native), counts)
    delta = r + discounts * bootstrap * vn - v
    advantages = np.empty_like(r)
    following = np.zeros(r.shape[1], dtype=np.float64)
    for k in range(r.shape[0] - 1, -1, -1):
        following = delta[k] + decay[k] * trace[k] * following
        advantages[k] = following
    targets = advantages + v
    if not np.isfinite(advantages).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite advantages or targets')
    return advantages, targets
