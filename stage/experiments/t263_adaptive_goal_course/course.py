"""Training-episode goal schedule only; no within-episode target changes."""
from copy import deepcopy

LEVELS = (5.25, 5.5, 5.75, 6.)


def advance_level(level, successful_draws, center_safe, retained_updates):
    assert 0 <= level < 4 and 0 <= successful_draws <= 8 and retained_updates >= 0
    if retained_updates > 0 and center_safe and successful_draws >= 2:
        return min(3, level + 1)
    if not center_safe and successful_draws == 0:
        return max(0, level - 1)
    return level


def course_job(job, bases, level):
    side = job['case']['side']
    base = bases[(side, level)]
    result = deepcopy(job)
    name = f'course{LEVELS[level]:.2f}_' + job['id']
    result.update(id=name, source_reset_id=name, case=deepcopy(base['case']),
                  exact_snapshot=base['exact_snapshot'], group='course_' + job['group'])
    result.pop('snapshot_destination', None)
    return result


def contracts():
    assert advance_level(0, 2, True, 1) == 1
    assert advance_level(0, 8, True, 0) == 0
    assert advance_level(1, 8, False, 4) == 1
    assert advance_level(2, 0, False, 1) == 1
    assert advance_level(0, 0, False, 0) == 0
    assert advance_level(3, 8, True, 4) == 3
    assert advance_level(2, 1, True, 4) == 2
    job = dict(id='dev6_LEFT_13', case=dict(side='LEFT'), group='6m_LEFT',
               delta_x_m=0., delta_y_m=0., delta_yaw_deg=0.)
    before = deepcopy(job)
    bases = {('LEFT', 1): dict(case=dict(side='LEFT', distance_m=5.5, step_cap=25667),
                              exact_snapshot='registered_frozen_snapshot')}
    result = course_job(job, bases, 1)
    assert job == before and result['case']['distance_m'] == 5.5
    assert result['exact_snapshot'] == 'registered_frozen_snapshot'
    assert result['delta_x_m'] == result['delta_y_m'] == result['delta_yaw_deg'] == 0
    return dict(advance_requires_actual_update_center_and_two_successes=True,
                deterministic_declared_fallback=True, source_jobs_unchanged=True,
                episode_goal_fixed=True, navigation_steps=0)
