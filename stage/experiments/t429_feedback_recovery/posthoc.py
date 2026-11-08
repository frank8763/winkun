"""Post-completion, zero-native/optimizer analysis of the sole failed recovery.

This file was added AFTER T429 completed. It is not a registered intervention
and never imports the simulator, learner, or T429 runtime.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'stage/runs/t429_feedback_recovery'
DEST = ROOT / 'stage/docs/feedback_20261108/FAILED_CLEARANCE_POSTHOC.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not DEST.exists(), 'Preserve the first posthoc output.'
    p = json.loads((RUN / 'protocol.json').read_text())
    assert json.loads((RUN / 'native/audit.json').read_text())['state'] == 'PASS'
    target = next(t for t in p['recoveries'] if t['job_id'] == 'dev_task6_21')
    rows = [json.loads(line) for line in (RUN / 'native/cases.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['job_id'] == target['job_id'])
    assert row['complete'] and row['collision'] and not row['success']
    paths = [row['path'], target['reference_path'], target['failed_mixed_row']['path']]
    hashes = [row['sha256'], target['reference_sha256'], target['failed_mixed_row']['sha256']]
    assert all(sha(path) == expected for path, expected in zip(paths, hashes))
    a, b, old = [np.load(path, allow_pickle=False) for path in paths]
    first = target['first']
    n = len(a['actions'])
    reference_poses = np.concatenate([b['poses'], b['next_poses'][-1:]], axis=0)
    wrap = lambda value: (value + np.pi) % (2 * np.pi) - np.pi
    clock = np.r_[0, np.cumsum(a['native_counts'].astype(np.int64))] * .002
    reference_clock = np.r_[0, np.cumsum(b['native_counts'].astype(np.int64))] * .002
    assert clock[first] == reference_clock[first] == 22.
    gains = np.asarray(p['gains'], np.float64)
    samples = []
    for i in range(first, n):
        j = int(a['feedback_progress'][i])
        assert 0 <= j < len(b['actions'])
        pose = a['poses'][i]
        target_pose = reference_poses[j]
        delta = target_pose[:2] - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        forward, left = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        yaw = wrap(target_pose[2] - pose[2])
        speed, target_speed = float(a['states'][i, 56]), float(b['states'][j, 56])
        base = b['actions'][j].astype(np.float64)
        goal_distance = np.linalg.norm(a['task_goal_xy'] - pose[:2])
        raw = np.array([
            base[0] + gains[2] * forward + gains[3] * (target_speed - speed)
            - gains[4] * abs(yaw) - gains[5] * max(0., 1. - goal_distance / .6),
            base[1] + gains[0] * left + gains[1] * yaw])
        assert np.array_equal(np.clip(raw, -1., 1.).astype(np.float32), a['actions'][i])
        samples.append(dict(time_s=float(clock[i]), reference_time_s=float(reference_clock[j]),
                            forward_error_m=float(forward), left_error_m=float(left),
                            yaw_error_deg=float(np.rad2deg(yaw)), speed_mps=speed,
                            reference_speed_mps=target_speed, executed=a['actions'][i].tolist()))
    actions = a['actions'][first:]
    out = dict(
        state='POSTHOC_NUMERICAL_EVIDENCE_NOT_A_NEW_INTERVENTION',
        new_native_steps=0, new_optimizer_calls=0, source_sha256=sha(__file__),
        inputs=[dict(path=path, sha256=value) for path, value in zip(paths, hashes)],
        handoff_seconds=float(clock[first]),
        same_clock_handoff_position_error_m=float(np.linalg.norm(a['poses'][first, :2] - reference_poses[first, :2])),
        same_clock_handoff_yaw_error_deg=float(np.rad2deg(wrap(reference_poses[first, 2] - a['poses'][first, 2]))),
        handoff_actual_speed_mps=float(a['states'][first, 56]),
        handoff_reference_speed_mps=float(b['states'][first, 56]),
        contact_time_s=float(clock[-1]),
        previous_neural_teacher_contact_time_s=float(old['native_counts'].astype(np.int64).sum() * .002),
        suffix_decisions=len(actions),
        positive_drive_saturation_fraction=float(np.mean(actions[:, 0] >= .9999)),
        positive_steer_saturation_fraction=float(np.mean(actions[:, 1] >= .9999)),
        negative_steer_saturation_fraction=float(np.mean(actions[:, 1] <= -.9999)),
        speed_min_median_max_mps=np.quantile(a['states'][first:, 56], [0, .5, 1]).tolist(),
        coordinate_definition='poses=[world x,world y,yaw radians]; positive left error in actual vehicle frame',
        samples=samples, all_suffix_feedback_actions_recomputed_exact=True,
        limitations=['No vehicle clearance or physical recoverability proof.',
                     'Same-clock handoff error differs from progress-matched feedback error.',
                     'Later collision time alone does not establish improvement.',
                     'No new actor, recovery attempt, or navigation acceptance.'])
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in out.items() if key not in ['inputs', 'samples', 'limitations']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
