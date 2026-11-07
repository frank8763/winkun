"""Training-only feedback around one authenticated successful Stage rollout.

This module never advances a simulator or updates a model. Its controls have no
neural teacher latent or hidden state. New native rollouts must record their own
actual observations, causal histories, executed actions and student hidden.
"""
from pathlib import Path
import hashlib
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def load_reference(path, expected_sha):
    """Read the existing lossless payload without creating output files.

    Reference times and poses include the final endpoint at index ``n``.
    Indices ``0..n-1`` correspond to the actual recorded action holds. Reaching
    endpoint index ``n`` switches feedback to the true task goal.
    """
    path = Path(path)
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, (str(path), actual_sha, expected_sha)
    directory = str(ROOT / 'stage/experiments/t418_evening_repair_chain')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import repair_core as E

    arrays = E.N.S.arrays(path)
    n = len(arrays['actions'])
    assert n > 0 and arrays['states'].shape == (n, 120)
    assert arrays['actions'].shape == (n, 2)
    assert arrays['poses'].shape == arrays['next_poses'].shape == (n, 3)
    assert arrays['native_counts'].shape == (n,)
    assert np.all(arrays['native_counts'] > 0)
    assert np.all(arrays['native_counts'] <= 100)
    assert not arrays['end_flags'][:-1].any()
    assert np.array_equal(arrays['end_flags'][-1], [0, 1, 0])
    assert arrays['history_poses'].shape == (n + 1, 4, 3)
    assert arrays['history_goals'].shape == (n + 1, 4, 2)
    assert arrays['history_times'].shape == (n + 1, 4)
    assert arrays['initial_native_integration'].shape == (203,)
    assert arrays['final_native_integration'].shape == (203,)
    assert np.array_equal(arrays['states'][1:], arrays['next_states'][:-1])
    assert np.array_equal(arrays['poses'][1:], arrays['next_poses'][:-1])
    goal = np.asarray(arrays['history_goals'][-1, -1], np.float64).copy()
    assert np.isfinite(goal).all()
    assert np.array_equal(arrays['history_goals'],
                          np.broadcast_to(goal, arrays['history_goals'].shape))
    if 'task_goal_xy' in arrays:
        assert np.array_equal(goal, arrays['task_goal_xy'])
    poses = np.concatenate([arrays['poses'], arrays['next_poses'][-1:]], axis=0)
    times = np.r_[0, np.cumsum(arrays['native_counts'].astype(np.int64))] * .002
    refspeed = np.r_[arrays['states'][:, 56].astype(np.float64), .025]
    assert np.isfinite(poses).all() and np.isfinite(refspeed).all()
    assert np.all(refspeed >= 0) and np.all(np.diff(times) > 0)
    assert np.linalg.norm(poses[-1, :2] - goal) <= .20
    return dict(arrays=arrays, times=times, poses=poses, refspeed=refspeed,
                goal=goal, source_path=str(path), source_sha256=actual_sha,
                native_timestep_s=.002, action_holds=n,
                final_time_s=float(times[-1]), training_only=True,
                teacher_latent_available=False)


def feedback(ref, pose, speed, elapsed_s, progress, gains, mode):
    """Return an executed-command proposal and explicit feedback diagnostics.

    ``time`` uses the real episode clock. ``progress`` matches only within the
    fixed forward index window ``progress..progress+12`` and cannot jump to a
    remote part of the reference. Neither mode resets native state or time.
    Gains are [ky, kpsi, kx, kv, kyawslow, kendslow].
    """
    pose = np.asarray(pose, np.float64)
    gains = np.asarray(gains, np.float64)
    assert pose.shape == (3,) and gains.shape == (6,)
    assert np.isfinite(pose).all() and np.isfinite(gains).all()
    speed, elapsed_s = float(speed), float(elapsed_s)
    assert np.isfinite(speed) and np.isfinite(elapsed_s)
    assert elapsed_s >= 0
    n = len(ref['arrays']['actions'])
    assert isinstance(progress, (int, np.integer)) and 0 <= progress <= n
    assert ref['poses'].shape == (n + 1, 3)
    assert ref['times'].shape == ref['refspeed'].shape == (n + 1,)
    assert mode in ('time', 'progress')
    if mode == 'time':
        index = int(np.clip(np.searchsorted(ref['times'], elapsed_s,
                                            side='right') - 1, 0, n))
        match_cost = None
    else:
        indices = np.arange(progress, min(n, progress + 12) + 1)
        delta = ref['poses'][indices, :2] - pose[:2]
        costs = np.square(delta).sum(axis=1) + .04 * np.square(
            wrap(ref['poses'][indices, 2] - pose[2]))
        choice = int(np.argmin(costs))
        index = int(indices[choice])
        match_cost = float(costs[choice])
    goal_delta = np.asarray(ref['goal'], np.float64) - pose[:2]
    goal_distance = float(np.linalg.norm(goal_delta))
    terminal = bool(index == n or (mode == 'time' and elapsed_s >= ref['times'][-1]))
    if terminal:
        target_xy = np.asarray(ref['goal'], np.float64)
        target_yaw = float(np.arctan2(goal_delta[1], goal_delta[0]))
        target_speed = .025
        base_action = np.zeros(2, np.float64)
    else:
        target_xy = ref['poses'][index, :2]
        target_yaw = float(ref['poses'][index, 2])
        target_speed = float(ref['refspeed'][index])
        base_action = ref['arrays']['actions'][index].astype(np.float64)
    delta = target_xy - pose[:2]
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    forward = float(cosine * delta[0] + sine * delta[1])
    left = float(-sine * delta[0] + cosine * delta[1])
    yaw_error = float(wrap(target_yaw - pose[2]))
    ky, kpsi, kx, kv, kyawslow, kendslow = gains
    steer = base_action[1] + ky * left + kpsi * yaw_error
    drive = (base_action[0] + kx * forward + kv * (target_speed - speed)
             - kyawslow * abs(yaw_error)
             - kendslow * max(0., 1. - goal_distance / .6))
    raw = np.asarray([drive, steer], np.float64)
    assert np.isfinite(raw).all()
    action = np.clip(raw, -1., 1.).astype(np.float32)
    diagnostic = dict(mode=mode, reference_index=index,
                      previous_progress=int(progress), match_cost=match_cost,
                      elapsed_s=elapsed_s, reference_time_s=float(ref['times'][index]),
                      terminal_goal_feedback=terminal,
                      target_xy=target_xy.tolist(), target_yaw=target_yaw,
                      target_speed=target_speed, base_action=base_action.tolist(),
                      error_forward_m=forward, error_left_m=left,
                      error_yaw_rad=yaw_error, speed_mps=speed,
                      goal_distance_m=goal_distance,
                      unclipped_action=raw.tolist(), executed_action=action.tolist(),
                      training_only=True, teacher_latent_available=False)
    return action, index, diagnostic


def pure_checks():
    """Synthetic sign/bounds/locality checks: zero files/native/optimizer work."""
    ref = dict(arrays=dict(actions=np.zeros((20, 2), np.float32)),
               times=np.arange(21, dtype=np.float64) * .2,
               poses=np.c_[np.arange(21) * .1, np.zeros(21), np.zeros(21)],
               refspeed=np.full(21, .1), goal=np.asarray([2.1, 0.]))
    gains = np.asarray([1., 1., 1., 1., .1, .1])
    aligned, index, _ = feedback(ref, [0., 0., 0.], .1, 0., 0, gains, 'time')
    assert index == 0 and np.array_equal(aligned, [0., 0.])
    left, _, dl = feedback(ref, [0., -.2, 0.], .1, 0., 0, gains, 'time')
    right, _, dr = feedback(ref, [0., .2, 0.], .1, 0., 0, gains, 'time')
    assert left[1] > 0 > right[1] and dl['error_left_m'] > 0 > dr['error_left_m']
    behind, _, _ = feedback(ref, [-.2, 0., 0.], .1, 0., 0, gains, 'time')
    ahead, _, _ = feedback(ref, [.2, 0., 0.], .1, 0., 0, gains, 'time')
    assert behind[0] > 0 > ahead[0]
    turn, _, dy = feedback(ref, [0., 0., -.2], .1, 0., 0, gains, 'time')
    assert turn[1] > 0 and dy['error_yaw_rad'] > 0
    bounded, _, _ = feedback(ref, [-100., -100., -2.], 0., 0., 0,
                            np.ones(6) * 100, 'time')
    assert bounded.dtype == np.float32 and np.abs(bounded).max() <= 1
    _, local, _ = feedback(ref, [2., 0., 0.], .1, 0., 0, gains, 'progress')
    assert local == 12
    _, monotone, _ = feedback(ref, [0., 0., 0.], .1, 0., 12, gains, 'progress')
    assert 12 <= monotone <= 20
    _, terminal_index, end = feedback(ref, [2., -.2, np.pi / 2], 0., 4.1,
                                      19, gains, 'time')
    assert terminal_index == 20 and end['terminal_goal_feedback']
    assert end['target_xy'] == ref['goal'].tolist()
    assert abs(float(wrap(end['target_yaw'] - np.arctan2(.2, .1)))) < 1e-12
    assert end['target_speed'] == .025 and end['base_action'] == [0., 0.]
    assert not end['teacher_latent_available']
    return dict(state='PASS', native=0, optimizer=0, file_writes=0,
                feedback_signs=True, bounded_float32=True,
                local_forward_window_12_and_monotone=True,
                true_clock_time_mode=True, terminal_aims_at_actual_goal=True,
                no_fabricated_teacher_latent=True)
