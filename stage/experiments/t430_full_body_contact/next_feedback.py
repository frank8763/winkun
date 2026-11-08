"""POSTHOC proposal, not a registered or natively tested controller.

Training-time feedback only. Uses a front-axle error and a softened low-speed
heading correction inspired by Stanley. This is not the original Stanley law
or its stability proof: it retains the existing feedforward/drive behavior and
uses its heading gain after an explicit radians/normalized-action conversion.
"""
import math
import numpy as np

FRONT_AXLE_X_M = .30
MAX_CENTER_STEER_RAD = math.radians(25.)
CROSS_TRACK_RATE_PER_S = 1.
SPEED_SOFTENING_MPS = .10


def wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def front_point(pose):
    x, y, yaw = np.asarray(pose, np.float64)
    return np.array([x + FRONT_AXLE_X_M * math.cos(yaw),
                     y + FRONT_AXLE_X_M * math.sin(yaw)])


def proposal(actual_pose, reference_pose, actual_speed_mps, base_action,
             existing_drive_action, existing_heading_gain, terminal=False,
             existing_terminal_steer=None):
    """One fixed proposal; no model, filesystem, environment or optimizer.

    positive cross-track error means the reference front axle is left of the
    actual one, measured in reference heading coordinates. The nonlinear
    heading offset should request a temporary heading beyond the reference,
    so a persistent lateral offset does not merely become a parallel path.
    Runtime continuation and obstacle clearance remain untested.
    """
    pose = np.asarray(actual_pose, np.float64)
    ref = np.asarray(reference_pose, np.float64)
    base = np.asarray(base_action, np.float64)
    assert pose.shape == ref.shape == (3,) and base.shape == (2,)
    assert np.isfinite(pose).all() and np.isfinite(ref).all() and np.isfinite(base).all()
    assert np.abs(base).max() <= 1, 'Reference feedforward must be a normalized executed command.'
    speed = float(actual_speed_mps)
    drive = float(existing_drive_action)
    gain = float(existing_heading_gain)
    assert math.isfinite(speed) and speed >= 0 and -1 <= drive <= 1
    assert math.isfinite(gain) and gain > 0
    if terminal:
        assert existing_terminal_steer is not None
        steer = float(existing_terminal_steer)
        assert -1 <= steer <= 1
        return np.asarray([drive, steer], np.float32), dict(terminal_unchanged=True)
    delta = front_point(ref) - front_point(pose)
    left = float(-math.sin(ref[2]) * delta[0] + math.cos(ref[2]) * delta[1])
    heading_offset = math.atan2(CROSS_TRACK_RATE_PER_S * left, speed + SPEED_SOFTENING_MPS)
    yaw_error = wrap(ref[2] - pose[2])
    desired_heading_error = wrap(yaw_error + heading_offset)
    # Existing gain has units normalized-command/radian. Multiplying by the
    # physical steering scale yields the corresponding dimensionless gain.
    physical_heading_gain = gain * MAX_CENTER_STEER_RAD
    feedforward_angle = float(base[1]) * MAX_CENTER_STEER_RAD
    raw_angle = feedforward_angle + physical_heading_gain * desired_heading_error
    center_angle = float(np.clip(raw_angle, -MAX_CENTER_STEER_RAD, MAX_CENTER_STEER_RAD))
    steer = center_angle / MAX_CENTER_STEER_RAD
    action = np.asarray([drive, steer], np.float32)
    assert np.isfinite(action).all() and np.abs(action).max() <= 1
    return action, dict(front_cross_track_error_m=left, body_yaw_error_rad=yaw_error,
        added_heading_rad=heading_offset, desired_heading_error_rad=desired_heading_error,
        actual_speed_mps=speed,
        feedforward_center_angle_rad=feedforward_angle, proposed_center_angle_rad=center_angle,
        unclipped_center_angle_rad=raw_angle, terminal_unchanged=False,
        training_only=True, native_rollout_tested=False)


def contract_checks():
    """Meaningful units/sign/bounds checks; not navigation validation."""
    gain = 1.3115789177481325
    zero = np.zeros(3)
    base = np.array([.3, .2])
    aligned, diag = proposal(zero, zero, .1, base, .3, gain)
    assert np.array_equal(aligned, base.astype(np.float32))
    assert diag['added_heading_rad'] == 0
    left, _ = proposal(zero, [0, .06, 0], .1, [0, 0], .4, gain)
    right, _ = proposal(zero, [0, -.06, 0], .1, [0, 0], .4, gain)
    fast, _ = proposal(zero, [0, .06, 0], .3, [0, 0], .4, gain)
    assert left[1] > 0 and right[1] < 0 and abs(fast[1]) < abs(left[1])
    assert np.isclose(left[1], -right[1]) and left[0] == right[0]
    stopped, _ = proposal(zero, [0, .06, 0], 0, [0, 0], .4, gain)
    assert np.isfinite(stopped).all()
    terminal, _ = proposal(zero, [1, 1, 0], .1, [0, 0], .4, gain,
                            terminal=True, existing_terminal_steer=-.25)
    assert np.array_equal(terminal, np.array([.4, -.25], np.float32))
    assert np.allclose(front_point([1, 2, math.pi / 2]), [1, 2.3])
    for action in [aligned, left, right, fast, stopped, terminal]:
        angle = float(action[1]) * MAX_CENTER_STEER_RAD
        assert abs(angle) <= MAX_CENTER_STEER_RAD
        assert np.isclose(angle / MAX_CENTER_STEER_RAD, action[1])
    return dict(state='PURE_CONTRACT_CHECKS_PASS', native_steps=0, Actor_Adam=0,
                no_registered_or_started_navigation=True,
                checks=['zero-error feedforward', 'left/right sign', 'low-speed scaling',
                        'zero-speed finite', 'terminal unchanged', 'front axle location',
                        'physical angle/normalized command roundtrip'])


if __name__ == '__main__':
    print(contract_checks())
