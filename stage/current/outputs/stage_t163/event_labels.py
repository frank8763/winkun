"""Finite-horizon first-event labels for a FIXED behavior policy episode.

Labels are [no event yet, goal first, collision first]. Unknown labels are -1.
Timeout/budget/unknown termination censor the observation, rather than prove safety.
These labels are not counterfactual action risks and not off-policy policy evaluation.
"""
import numpy as np

def first_event_labels(sample_times_s, end_time_s, end_kind, horizons_s):
    times = np.asarray(sample_times_s, dtype=np.float64)
    horizons = np.asarray(horizons_s, dtype=np.float64)
    assert times.ndim == horizons.ndim == 1
    assert np.isfinite(times).all() and np.isfinite(horizons).all()
    assert np.isfinite(end_time_s) and np.all(horizons > 0)
    remaining = float(end_time_s) - times
    assert np.all(remaining >= -1e-9)
    assert end_kind in ('GOAL', 'COLLISION', 'STEP_CAP', 'TRUNCATED', 'BUDGET', 'TERMINATED')
    complete_event = end_kind in ('GOAL', 'COLLISION')
    observed_horizon = horizons[None, :] <= remaining[:, None] + 1e-9
    mask = observed_horizon | complete_event
    labels = np.where(mask, 0, -1).astype(np.int8)
    if complete_event:
        event_inside = remaining[:, None] <= horizons[None, :] + 1e-9
        labels[event_inside] = 1 if end_kind == 'GOAL' else 2
    return labels, mask

def check_boundary_cases():
    y, m = first_event_labels([0., .8, 1.], 1., 'COLLISION', [.2, 1., 3.])
    assert y.tolist() == [[0, 2, 2], [2, 2, 2], [2, 2, 2]] and m.all()
    y, m = first_event_labels([0., .8, 1.], 1., 'TRUNCATED', [.2, 1., 3.])
    assert y.tolist() == [[0, 0, -1], [0, -1, -1], [-1, -1, -1]]
    y, m = first_event_labels([0., .8], 1., 'GOAL', [.2, 1., 3.])
    assert y.tolist() == [[0, 1, 1], [1, 1, 1]] and m.all()
    assert not np.any(y == 2)
    print('PASS: event boundary, censoring, competing terminal event, absorbing goal')

if __name__ == '__main__': check_boundary_cases()
