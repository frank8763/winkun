"""Independent bounded evidence/Adam audit of the actual-recovery pilot.

No native integration is allowed. The front-axle controller is reconstructed
algebraically without importing or calling its producer. Learn audit performs
exactly the registered240 discarded historical Adam steps, not new training.
"""
from pathlib import Path
import copy
import fcntl
import importlib.util
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('stage431_pinned427_audit',
    ROOT / 'stage/experiments/t427_fixed_representation/audit.py')
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)
H = O.T
sha, jsonl = H.sha, H.jsonl


def authenticated(C, p, row, guard, check_policy=True):
    guard()
    assert row['path'] in p['inputs'] and p['inputs'][row['path']] == row['sha256'] == sha(row['path'])
    a = H.checked_array(C, row)
    assert row['full'] and set(a) == set(row['array_hashes'])
    if check_policy:
        O.full_policy(C, a, row, p)
    return a


def actual_prefix(C, source, mixed, first):
    """Every original recorded prefix field, including causal histories."""
    E = C.E
    n, m = len(source['actions']), len(mixed['actions'])
    assert 0 <= first < n and first < m
    assert set(mixed) == set(source) | {'feedback_progress', 'feedback_speed'}
    constants = ['initial_native_integration', 'task_goal_xy', 'scene_sha256']
    histories = ['history_poses', 'history_goals', 'history_times']
    terminal = ['final_native_integration', 'last_hidden_after']
    for key, value in source.items():
        if key in constants:
            assert E.equal(value, mixed[key]), key
        elif key in histories:
            assert value.shape[0] == n + 1 and mixed[key].shape[0] == m + 1
            assert E.equal(value[:first + 1], mixed[key][:first + 1]), key
        elif key not in terminal:
            assert value.shape[0] == n and mixed[key].shape[0] == m
            assert E.equal(value[:first], mixed[key][:first]), key
    for key in ['states', 'poses', 'hidden_before']:
        assert E.equal(source[key][:first + 1], mixed[key][:first + 1]), ('handoff', key)
    assert not mixed['ownership'][:first].any() and mixed['ownership'][first:].all()
    assert not mixed['stochastic'].any()


def independent_front_controls(C, p, arrays, row):
    """Recompute progress/drive/front steering without producer functions."""
    np, E = C.np, C.E
    request = row['request']
    assert request['mode'] == 'front_feedback' and request['seed'] is None
    first = request['first']
    gains = np.asarray(request['gains'], np.float64)
    assert gains.shape == (6,) and gains.tolist() == p['gains']
    reference_path = request['reference_path']
    assert p['inputs'][reference_path] == request['reference_sha256'] == sha(reference_path)
    ref = E.N.S.arrays(reference_path)
    assert np.array_equal(ref['end_flags'][-1], [0, 1, 0]) and not ref['end_flags'][:-1].any()
    assert np.array_equal(ref['initial_native_integration'], arrays['initial_native_integration'])
    assert np.array_equal(ref['task_goal_xy'], arrays['task_goal_xy'])
    assert np.array_equal(ref['scene_sha256'], arrays['scene_sha256'])
    n = len(ref['actions'])
    poses = np.concatenate([ref['poses'], ref['next_poses'][-1:]])
    speed_ref = np.r_[ref['states'][:, 56].astype(np.float64), .025]
    goal = np.asarray(ref['task_goal_xy'], np.float64)
    progress = first
    max_angle = float(np.deg2rad(25.))
    wrap = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi
    assert np.all(arrays['feedback_progress'][:first] == -1)
    assert np.all(arrays['feedback_speed'][:first] == 0)
    assert np.array_equal(arrays['feedback_speed'][first:], arrays['states'][first:, 56])
    for i in range(first, len(arrays['actions'])):
        pose = arrays['poses'][i]
        current_speed = float(arrays['states'][i, 56])
        indices = np.arange(progress, min(n, progress + 12) + 1)
        difference = poses[indices, :2] - pose[:2]
        costs = np.square(difference).sum(1) + .04 * np.square(wrap(poses[indices, 2] - pose[2]))
        progress = int(indices[int(np.argmin(costs))])
        assert progress == int(arrays['feedback_progress'][i])
        goal_delta = goal - pose[:2]
        distance = float(np.linalg.norm(goal_delta))
        terminal = progress == n
        if terminal:
            target_xy = goal
            target_yaw = float(np.arctan2(goal_delta[1], goal_delta[0]))
            target_speed = .025
            base = np.zeros(2, np.float64)
        else:
            target_xy = poses[progress, :2]
            target_yaw = float(poses[progress, 2])
            target_speed = float(speed_ref[progress])
            base = ref['actions'][progress].astype(np.float64)
        delta = target_xy - pose[:2]
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        forward = float(cosine * delta[0] + sine * delta[1])
        left = float(-sine * delta[0] + cosine * delta[1])
        yaw = float(wrap(target_yaw - pose[2]))
        ky, kpsi, kx, kv, kyawslow, kendslow = gains
        drive = float(np.clip(base[0] + kx * forward + kv * (target_speed - current_speed)
                      - kyawslow * abs(yaw) - kendslow * max(0., 1. - distance / .6), -1., 1.))
        if terminal:
            steer = float(np.clip(base[1] + ky * left + kpsi * yaw, -1., 1.))
        else:
            # math-library scalar trig follows the producer's scalar
            # representation; the formula is independently expressed here.
            import math
            front_actual = np.asarray([pose[0] + .30 * math.cos(pose[2]),
                                       pose[1] + .30 * math.sin(pose[2])])
            front_target = np.asarray([poses[progress, 0] + .30 * math.cos(target_yaw),
                                       poses[progress, 1] + .30 * math.sin(target_yaw)])
            front_delta = front_target - front_actual
            cross = float(-math.sin(target_yaw) * front_delta[0] + math.cos(target_yaw) * front_delta[1])
            offset = math.atan2(cross, current_speed + .10)
            desired = (yaw + offset + math.pi) % (2 * math.pi) - math.pi
            raw_angle = float(base[1]) * max_angle + (float(kpsi) * max_angle) * desired
            steer = float(np.clip(raw_angle, -max_angle, max_angle)) / max_angle
        expected = np.asarray([drive, steer], np.float32)
        assert E.equal(expected, arrays['actions'][i]), (row['job_id'], i, 'independent_front_control')
    return len(arrays['actions']) - first


def full_policy(C, p, arrays, row):
    if row['mode'] != 'front_feedback':
        O.full_policy(C, arrays, row, p)
        return
    q = C.frozen(p, row['request']['model']).requires_grad_(False)
    assert q.phash() == row['policy_hash']
    before = C.digest(q.state_dict())
    C.E.audit_array(q, arrays, p, row['request']['first'], None)
    independent_front_controls(C, p, arrays, row)
    assert C.digest(q.state_dict()) == before


def same_replay(row, reference):
    assert row['request'] == reference['request'] and row['reference_path'] == reference['path']
    assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
    for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics_steps',
                'success', 'collision', 'timeout', 'policy_hash', 'loaded_model_sha256', 'mode', 'first']:
        assert row[key] == reference[key], key


def checked_existing_dataset(C, p, dataset, guard):
    """All existing labels/anchors are authentic, audited, current-policy data."""
    index_path = Path(p['recovery_support_index_path'])
    assert p['recovery_support_index_sha256'] == p['inputs'][str(index_path)] == sha(index_path)
    index = C.read(index_path)
    six = index['successes']
    assert len(six) == 6 and len({r['job_id'] for r in six}) == 6
    targets = dataset['targets']
    target_map = {entry['id']: entry for entry in targets}
    assert len(target_map) == len(targets) and len(targets) in [6, 7]
    for source in six:
        guard()
        entry = target_map[source['job_id']]
        assert entry['path'] == source['path'] and entry['sha256'] == source['sha256']
        assert entry['first'] == source['first_feedback_or_teacher_hold']
        assert entry['row']['path'] == entry['path'] and entry['row']['sha256'] == entry['sha256']
        assert entry['request'] == entry['row']['request'] and entry['exact'] is True
        assert entry['row']['policy_hash'] == p['expected_policy_hash']
        assert source['policy_hash'] == p['expected_policy_hash']
        assert source['on_policy_for_PPO'] is False
        audit_path = source['source_audit_path']
        assert sha(audit_path) == source['source_audit_sha256'] == p['inputs'][audit_path]
        assert C.read(audit_path)['state'] == 'PASS'
        a = authenticated(C, p, entry['row'], guard)
        assert not a['ownership'][:entry['first']].any() and a['ownership'][entry['first']:].all()
        assert a['end_flags'][-1, 1] and not a['end_flags'][:, 0].any()
    source_cases = Path(p['current_cases_path'])
    assert p['inputs'][str(source_cases)] == sha(source_cases)
    source_rows = [r for r in jsonl(source_cases) if r['full'] and r['success'] and not r['collision']]
    anchors = dataset['anchors']
    assert len(anchors) == len(source_rows) == 59
    assert {entry['row']['job_id'] for entry in anchors} == {r['job_id'] for r in source_rows}
    by_id = {r['job_id']: r for r in source_rows}
    for entry in anchors:
        guard()
        row = entry['row']
        assert row == by_id[row['job_id']]
        assert entry['path'] == row['path'] and entry['sha256'] == row['sha256']
        assert row['policy_hash'] == p['expected_policy_hash']
        a = authenticated(C, p, row, guard)
        assert not a['ownership'].any() and a['end_flags'][-1, 1]
    assert len({r['row']['job_id'] for r in anchors} & {j['id'] for j in p['main_jobs']}) == 51
    assert len({r['row']['job_id'] for r in anchors} & {j['id'] for j in p['old8_jobs']}) == 8
    return six


def data_audit(C, p, out, summary, rows, replays, guard):
    result = summary['result']
    assert len(rows) == len(replays) == 1
    row = rows[0]
    assert row['role'] == 'FRONT_RECOVERY' and replays[0]['role'] == 'FRONT_EXACT'
    same_replay(replays[0], row)
    target = p['front_recovery']
    assert row['request']['job'] == target['job'] and row['request']['first'] == target['first'] == 110
    assert row['request']['reference_path'] == target['reference_path']
    assert row['request']['reference_sha256'] == target['reference_sha256']
    assert row['request']['model'] == p['student_start_model']
    assert row['loaded_model_sha256'] == p['student_start_sha256']
    assert row['policy_hash'] == p['expected_policy_hash']
    a = H.checked_array(C, row)
    original = authenticated(C, p, target['student_row'], guard)
    actual_prefix(C, original, a, target['first'])
    controls = independent_front_controls(C, p, a, row)
    assert C.read(out / 'anchors.json') == result['anchors']
    assert C.read(out / 'targets.json') == result['targets']
    source = result['source_model']
    assert source == C.initial_record(p)
    H.snapshot(C.E, source)
    six = checked_existing_dataset(C, p, result, guard)
    safe = bool(row['success'] and not row['collision'])
    assert len(result['targets']) == 6 + int(safe)
    existing_ids = {r['job_id'] for r in six}
    ids = {entry['id'] for entry in result['targets']}
    assert ids == existing_ids | ({target['job_id']} if safe else set())
    if safe:
        entry = next(entry for entry in result['targets'] if entry['id'] == target['job_id'])
        expected = dict(id=row['job_id'], path=row['path'], sha256=row['sha256'],
                        first=row['request']['first'], row=row, request=row['request'], exact=True)
        assert all(entry[key] == value for key, value in expected.items())
        assert entry['existing_audited_recovery'] is False
    assert result['front_recovery'] == row
    proof = result['front_prefix_check']
    assert C.read(out / 'front_prefix_check.json') == proof
    assert proof['first'] == target['first'] and proof['handoff_seconds'] == 22.
    assert proof['all_original_student_prefix_arrays_exact'] is True
    assert proof['handoff_state_hidden_history_exact'] is True
    assert proof['no_native_or_clock_or_history_reset'] is True
    assert result['learning_data_available'] is True
    assert result['front_recovery_safe'] == safe and result['front_exact_replay'] is True
    return dict(historical_discarded_Adam=0, existing_safe_recovery_supports=6,
                verified_actual_recovery_supports=len(ids), new_front_recovery_safe=safe,
                new_front_feedback_decisions_recomputed=controls, exact_new_native_replays=1,
                actual_student_prefix_history_hidden_and_clock_exact=True,
                authentic_current_policy_success_anchor_histories=59,
                source_successes_are_not_new_independent_samples=True)


def learn_audit(C, p, out, summary, rows, replays, guard):
    E, L = C.E, C.LEARNING
    result = summary['result']
    dataset = C.read(C.OUT / 'data/summary.json')['result']
    assert C.read(C.OUT / 'data/audit.json')['state'] == 'PASS'
    assert dataset['learning_data_available'] is True and not replays
    checked_existing_dataset(C, p, dataset, guard)
    local = copy.deepcopy(p)
    local['recovery_target_ids'] = [entry['id'] for entry in dataset['targets']]
    q = C.frozen(p, result['initial_model']['path'])
    opt = L.make_optimizer(q, p)
    H.check_model(E, q, opt, result['initial_model'])
    assert not opt.state and q.phash() == p['expected_policy_hash']
    assert C.digest(q.state_dict()) == C.digest(C.frozen(p, p['student_start_model']).state_dict())
    assert result['initial_model'] == result['before_model']
    assert sum(v.numel() for v in L.params(q)) == 66
    fixed = L.fixedhash(q)
    anchors = [(E.N.S.arrays(x['path']), x['row']) for x in dataset['anchors']]
    targets = [dict(arrays=E.N.S.arrays(x['path']), first=x['first'], id=x['id']) for x in dataset['targets']]
    groups = L.groups(q, local, anchors, targets)
    events = jsonl(out / 'optimizer_calls.jsonl')
    assert len(events) == 480 and {r['tag'] for r in events} == {'fit120'}
    fit = lambda event: L.fit(q, opt, groups, local, guard, event, steps=120)
    rebuilt = H.reconstruct_pair(E, q, opt, fit, result['fit'], guard, events, 'fit120', C.gradient_sha)
    assert rebuilt['events'] == events
    H.check_model(E, q, opt, result['proposed_model'])
    assert L.fixedhash(q) == fixed
    check = C.read(result['check_file'])
    assert len(rows) == 329 and {r['role'] for r in rows} == {'PROPOSAL120'}
    ids, actual = H.complete_development(C, p, check, rows, 'PROPOSAL120', result['proposed_model'])
    assert actual == rows
    full_ids = {j['id'] for j in p['reload_jobs']} - {j['id'] for j in p['fresh_jobs']}
    assert len(full_ids) == 62 and {r['job_id'] for r in rows if r['full']} == full_ids
    initial = set(p['initial_success_ids'])
    assert len(initial) == 322
    source_rows = jsonl(p['current_cases_path'])
    assert initial == {r['job_id'] for r in source_rows if r['success'] and not r['collision']}
    accepted = initial <= ids and bool(ids - initial)
    assert result['accepted'] == accepted
    chosen = result['proposed_model'] if accepted else result['initial_model']
    E.L.load(q, opt, C.model_decode(chosen['path'])['state'])
    H.check_model(E, q, opt, result['model'])
    assert L.fixedhash(q) == fixed
    assert result['selected_success_ids'] == sorted(ids if accepted else initial)
    assert result['updates'] == dict(live=120, shadow=120)
    assert result['development_all_safe'] == bool(accepted and check['all_safe'])
    assert result['development_file'] == (result['check_file'] if accepted else None)
    return dict(historical_discarded_Adam=240, actual_recorded_live_steps=120,
                actual_recorded_shadow_steps=120, fresh_Adam_initialization_proven=True,
                recorded_Actor_Adam_and_gradient_events_all_exact=True,
                actual_current59_anchor_histories_and6_or7_recovery_masks_checked=True,
                complete329_acceptance_and_whole_Actor_Adam_rollback_checked=True,
                full62_references_saved_in_same_development=True,
                fixed_representation_all_but66_parameters_exact=True)


def eight_audit(C, p, out, summary, rows, replays, guard):
    result = summary['result']
    assert not replays
    assert C.read(C.OUT / 'gate/summary.json')['result']['six_gate'] is True
    assert C.read(C.OUT / 'gate/audit.json')['state'] == 'PASS'
    geometry_path = p['eight_geometry_path']
    assert p['inputs'][geometry_path] == sha(geometry_path)
    geometry = C.read(geometry_path)
    assert len(geometry) == len(p['eight_jobs']) == 56
    assert [g['id'] for g in geometry] == [j['id'] for j in p['eight_jobs']]
    valid = {g['id'] for g in geometry if g['valid']}
    assert [r['job_id'] for r in rows] == [j['id'] for j in p['eight_jobs'] if j['id'] in valid]
    assert result['invalid_geometry'] == [g for g in geometry if not g['valid']]
    assert all(r['role'] == 'EIGHT_DIAGNOSTIC' and r['seed'] is None and not r['teacher']
               and r['loaded_model_sha256'] == result['model']['sha256']
               and r['policy_hash'] == result['model']['policy_hash'] for r in rows)
    assert result['diagnostic'] == C.E.tally(rows)
    assert result['eight_started'] is True and result['eight_training'] is False
    assert result['seven_skipped'] and result['six_gate_previously_passed']
    assert result['not_eight_stable_acceptance'] is True
    return dict(historical_discarded_Adam=0, eight_frozen_diagnostic_only=True,
                exact_registered_geometry_input_verified=True,
                seven_skipped_by_user_authorization=True, no_eight_stability_claim=True)


def main(name):
    import t431_core as C
    assert name in ['data', 'learn', 'gate', 'eight']
    p = C.read(C.OUT / 'protocol.json')
    C.verify(p)
    out = C.OUT / name
    assert not (out / 'audit.json').exists()
    summary = C.read(out / 'summary.json')
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    lock = (ROOT / 'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        absent = H.stopped(out)
        members = H.manifest(ROOT, out)
        assert not list(out.rglob('partial*.npz'))
        start = time.time()
        budget = p['stage_budgets'][name]
        C.E.N.offline()
        C.torch.set_num_threads(1)
        C.torch.use_deterministic_algorithms(True)
        def guard():
            assert time.time() < min(start + budget['audit_seconds'] - 5, p['deadline_all'] - 5)
        def prohibited(*_a, **_kw):
            raise AssertionError('Independent T431 audit forbids native/exports')
        C.E.L.Session = prohibited
        C.Pool = prohibited
        C.E.Pool = prohibited
        C.E.N.S.P.C.B.setup = prohibited
        actual_adam = [0]
        original_adam = C.torch.optim.Adam.step
        def counted_adam(optimizer, *args, **kwargs):
            guard()
            assert name == 'learn' and actual_adam[0] < 240
            result = original_adam(optimizer, *args, **kwargs)
            actual_adam[0] += 1
            return result
        C.torch.optim.Adam.step = counted_adam
        C.torch.optim.AdamW.step = prohibited
        C.torch.optim.SGD.step = prohibited
        C.E.export = prohibited
        C.LEARNING.export_model = prohibited
        rows = jsonl(out / 'cases.jsonl')
        replays = jsonl(out / 'replays.jsonl')
        commands = jsonl(out / 'commands.jsonl')
        for row in rows:
            guard()
            arrays = H.checked_array(C, row)
            if row['full']:
                full_policy(C, p, arrays, row)
            assert row['pid'] in absent
        for row in replays:
            assert row['pid'] in absent and row['physics'] == row['physics_steps']
            assert row['complete'] and not row['administrative_censor']
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
        if name == 'data':
            details = data_audit(C, p, out, summary, rows, replays, guard)
        elif name == 'learn':
            details = learn_audit(C, p, out, summary, rows, replays, guard)
        elif name == 'gate':
            details = O.gate_audit(C, p, out, summary, rows, replays, guard)
        else:
            details = eight_audit(C, p, out, summary, rows, replays, guard)
        assert actual_adam[0] == details['historical_discarded_Adam']
        assert [r['id'] for r in commands] == list(range(1, len(commands) + 1))
        assert len(commands) == summary['commands'] == 4 + len(rows) + len(replays) <= budget['max_commands']
        assert [(r['role'], r['physics']) for r in commands[:4]] == [('SETUP', 0)] * 4
        expected = [(r['role'], r['physics'], r['pid']) for r in rows + replays]
        actual = [(r['role'], r['physics'], r['pid']) for r in commands[4:]]
        assert sorted(expected) == sorted(actual)
        assert sum(r['physics'] for r in commands) == summary['physics'] <= budget['max_physics']
        assert summary['elapsed_s'] <= budget['main_seconds']
        guard()
        C.verify(p)
        C.save(out / 'audit.json', dict(state='PASS', stage=name, native=0,
            new_navigation_training=0, elapsed_s=time.time() - start,
            native_processes_absent=absent, manifest_members_before_audit=members,
            physics=summary['physics'], commands=summary['commands'],
            raw_commands_physics_and_termination_ledger_exact=True,
            full_actual_history_hidden_densities_rewards_checked=True,
            compact_evidence_limitation='Compact cases retain actions/counts/flags and full hashes; absent120 arrays are not reconstructed by zero-native audit.',
            original_T422_training_audit_not_repaired=True, **details))
        C.E.N.manifest(out)
        print('T431_STAGE_AUDIT_PASS', name, flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    assert len(sys.argv) == 2
    try:
        main(sys.argv[1])
    except BaseException:
        # Preserve the failed audit without overwriting any original evidence.
        try:
            import t431_core as C
            target = C.OUT / sys.argv[1] / 'audit_error.json'
            if not target.exists():
                C.save(target, dict(state='FAILED', error=traceback.format_exc(),
                                   epoch=time.time(), no_native=True, no_restart=True))
        finally:
            raise
