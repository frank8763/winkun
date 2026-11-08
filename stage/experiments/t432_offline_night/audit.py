"""Independent zero-native audit for the bounded T432 overnight chain.

Import is inert. An arm audit reconstructs only its registered 120 live and
120 shadow historical Adam steps. It never exports a model or launches a
native worker. Compact development evidence cannot reconstruct absent120
arrays; the conditional gate obtains and checks its own62 full references.
"""
from pathlib import Path
import copy
import fcntl
import importlib.util
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('stage432_immutable431_audit',
    ROOT / 'stage/experiments/t431_actual_recovery/audit.py')
PREVIOUS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PREVIOUS)
H = PREVIOUS.H
sha, jsonl = H.sha, H.jsonl


def locked(C, p, path):
    path = str(path)
    assert path in p['inputs'] and sha(path) == p['inputs'][path], path
    return C.read(path)


def full_policy(C, p, arrays, row):
    """Historical student/teacher/front-feedback controls use pinned audit."""
    PREVIOUS.full_policy(C, p, arrays, row)


def authenticated(C, p, row, guard):
    guard()
    assert row['path'] in p['inputs']
    assert p['inputs'][row['path']] == row['sha256'] == sha(row['path'])
    arrays = H.checked_array(C, row)
    assert row['full'] and set(arrays) == set(row['array_hashes'])
    full_policy(C, p, arrays, row)
    return arrays


def same_physics(row, reference):
    for key in ['job_id', 'array_digest', 'array_hashes', 'integration_sha256',
                'physics_steps', 'success', 'collision', 'timeout', 'remaining_m',
                'policy_hash', 'loaded_model_sha256']:
        assert row[key] == reference[key], (row['job_id'], key)


def safe_student(row, source):
    assert row['full'] and row['success'] and not row['collision'] and not row['timeout']
    assert row['mode'] == 'student' and not row['teacher'] and row['seed'] is None
    assert row['policy_hash'] == source['policy_hash']
    assert row['loaded_model_sha256'] == source['sha256']


def checked_sources(C, p, guard):
    """Authenticate the7 already executed recoveries and59 actual anchors."""
    targets = locked(C, p, p['verified_target_file'])
    audit = locked(C, p, p['verified_target_audit'])
    assert audit['state'] == 'PASS' and len(targets) == 7
    assert len({entry['id'] for entry in targets}) == 7
    assert {entry['id'] for entry in targets} == set(p['recovery_target_ids'])
    source = C.initial_record(p)
    H.snapshot(C.E, source)
    for entry in targets:
        guard()
        row = entry['row']
        assert entry['id'] == row['job_id'] and entry['path'] == row['path']
        assert entry['sha256'] == row['sha256'] and entry['request'] == row['request']
        assert entry['exact'] is True and entry['first'] == row['request']['first']
        assert row['policy_hash'] == p['expected_policy_hash']
        assert row['loaded_model_sha256'] == p['student_start_sha256']
        assert row['success'] and not row['collision'] and not row['timeout']
        if 'source_audit_path' in entry:
            assert entry['source_audit_sha256'] == p['inputs'][entry['source_audit_path']]
            assert locked(C, p, entry['source_audit_path'])['state'] == 'PASS'
        arrays = authenticated(C, p, row, guard)
        first = entry['first']
        assert type(first) is int and first in p['allowed_teacher_first']
        assert 0 < first < len(arrays['actions'])
        assert not arrays['ownership'][:first].any() and arrays['ownership'][first:].all()
        assert not arrays['stochastic'].any()
        assert arrays['end_flags'][-1, 1] and not arrays['end_flags'][:, 0].any()
    source_rows = jsonl(p['current_cases_path'])
    assert sha(p['current_cases_path']) == p['inputs'][p['current_cases_path']]
    assert len(source_rows) == 329
    initial = {r['job_id'] for r in source_rows if r['success'] and not r['collision']}
    assert len(initial) == 322 and initial == set(p['initial_success_ids'])
    anchors = p['current_anchor_rows']
    assert len(anchors) == 59 and len({r['job_id'] for r in anchors}) == 59
    expected = [r for r in source_rows if r['full'] and r['success'] and not r['collision']]
    assert anchors == expected
    for row in anchors:
        guard()
        safe_student(row, source)
        arrays = authenticated(C, p, row, guard)
        assert not arrays['ownership'].any() and not arrays['stochastic'].any()
    old = {j['id'] for j in p['old8_jobs']}
    six = {j['id'] for j in p['main_jobs']}
    assert len({r['job_id'] for r in anchors} & old) == 8
    assert len({r['job_id'] for r in anchors} & six) == 51
    return targets, anchors


def data_audit(C, p, out, summary, rows, replays, guard):
    result = summary['result']
    targets, original = checked_sources(C, p, guard)
    source = C.initial_record(p)
    assert result['source_model'] == source
    assert len(rows) == len(replays) == 3
    jobs = p['data_jobs']
    proofs = p['data_proofs']
    assert jobs == p['additional_old_jobs']
    assert len(jobs) == len(proofs) == 3
    assert [r['job_id'] for r in rows] == [j['id'] for j in jobs]
    assert [r['job_id'] for r in proofs] == [j['id'] for j in jobs]
    assert not ({r['job_id'] for r in original} & {j['id'] for j in jobs})
    for row, replay, job, proof in zip(rows, replays, jobs, proofs):
        guard()
        safe_student(row, source)
        assert row['request']['job'] == job and row['request']['model'] == source['path']
        same_physics(row, proof)
        assert proof in jsonl(p['current_cases_path']) and not proof['full']
        PREVIOUS.same_replay(replay, row)
        assert replay['request'] == row['request']
        assert row['role'] == 'ADDITIONAL_OLD3' and replay['role'] == 'ADDITIONAL_OLD3_EXACT'
    anchors_b = [dict(path=r['path'], sha256=r['sha256'], row=r) for r in original]
    anchors_c = anchors_b + [dict(path=r['path'], sha256=r['sha256'], row=r) for r in rows]
    assert result['targets'] == targets == C.read(out / 'targets.json')
    assert result['anchors_b'] == anchors_b == C.read(out / 'anchors_b.json')
    assert result['anchors_c'] == anchors_c == C.read(out / 'anchors_c.json')
    assert result['additional_old3'] == anchors_c[59:]
    assert result['verified_recovery_targets'] == 7
    assert result['additional_old3_original_arrays_exact'] is True
    assert result['additional_old3_native_exact'] is True
    assert result['learning_data_available'] is True
    return dict(historical_discarded_Adam=0, actual_recovery_supports=7,
                authentic_existing_current_anchor_histories=59,
                extra_current_old_histories=3, exact_extra_native_replays=3,
                original_compact_source_full_array_hashes_and_integration_exact=True,
                existing_recovery_control_modes_independently_recomputed=True,
                reused_recoveries_are_not_new_independent_samples=True)


def checked_dataset(C, p, guard):
    folder = C.OUT / 'data'
    assert C.read(folder / 'summary.json')['state'] == 'COMPLETE'
    assert C.read(folder / 'audit.json')['state'] == 'PASS'
    H.stopped(folder)
    H.manifest(ROOT, folder)
    result = C.read(folder / 'summary.json')['result']
    targets, original = checked_sources(C, p, guard)
    assert result['targets'] == targets == C.read(folder / 'targets.json')
    assert len(result['anchors_b']) == 59 and len(result['anchors_c']) == 62
    assert [v['row'] for v in result['anchors_b']] == original
    assert result['anchors_c'][:59] == result['anchors_b']
    extra = jsonl(folder / 'cases.jsonl')
    assert [v['row'] for v in result['anchors_c'][59:]] == extra
    for entry in result['anchors_c']:
        guard()
        assert entry['path'] == entry['row']['path'] and entry['sha256'] == entry['row']['sha256']
        arrays = H.checked_array(C, entry['row'])
        safe_student(entry['row'], result['source_model'])
        assert not arrays['ownership'].any()
    return result


def arm_audit(C, p, name, out, summary, rows, replays, guard):
    E, L = C.E, C.LEARNING
    result = summary['result']
    dataset = checked_dataset(C, p, guard)
    assert not replays and name in ['b', 'c']
    if name == 'c':
        assert C.read(C.OUT / 'b/audit.json')['state'] == 'PASS'
        assert C.read(C.OUT / 'b/summary.json')['result']['development_all_safe'] is False
    local = copy.deepcopy(p)
    local['arm'] = name
    local['phase_arm'] = name
    local['recovery_target_ids'] = [v['id'] for v in dataset['targets']]
    q = C.frozen(p, result['initial_model']['path'])
    opt = L.make_optimizer(q, local)
    H.check_model(E, q, opt, result['initial_model'])
    assert result['initial_model'] == result['before_model']
    assert not opt.state and q.phash() == p['expected_policy_hash']
    assert C.digest(q.state_dict()) == C.digest(C.frozen(p, p['student_start_model']).state_dict())
    assert sum(v.numel() for v in L.params(q)) == 4450
    assert {n for n, v in q.named_parameters() if v.requires_grad} == {
        'extra.0.weight', 'extra.0.bias', 'extra.2.weight', 'extra.2.bias'}
    fixed = L.fixedhash(q)
    anchors = dataset['anchors_' + name]
    assert len(anchors) == (59 if name == 'b' else 62)
    jobs = {j['id']: j for j in p['main_jobs'] + p['old275_jobs']}
    local['anchor_jobs'] = [jobs[v['row']['job_id']] for v in anchors]
    pairs = [(E.N.S.arrays(v['path']), v['row']) for v in anchors]
    targets = [dict(arrays=E.N.S.arrays(v['path']), first=v['first'], id=v['id'])
               for v in dataset['targets']]
    groups = L.groups(q, local, pairs, targets)
    events = jsonl(out / 'optimizer_calls.jsonl')
    tag = 'fit120_' + name
    assert len(events) == 480 and {r['tag'] for r in events} == {tag}
    fit = lambda event: L.fit(q, opt, groups, local, guard, event, steps=120)
    rebuilt = H.reconstruct_pair(E, q, opt, fit, result['fit'], guard, events,
                                 tag, C.gradient_sha)
    assert rebuilt['events'] == events
    H.check_model(E, q, opt, result['proposed_model'])
    assert L.fixedhash(q) == fixed
    check = C.read(result['check_file'])
    assert len(rows) == 329 and all(not r['full'] for r in rows)
    ids, actual = H.complete_development(C, p, check, rows, 'PROPOSAL120_' + name, result['proposed_model'])
    assert actual == rows
    initial = set(p['initial_success_ids'])
    assert len(initial) == 322
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
    assert result['current_anchor_count'] == len(anchors)
    assert result['arm'] == name and result['trainable_parameters'] == 4450
    assert result['initial_source_policy_hash'] == p['expected_policy_hash']
    assert result['recovery_target_ids'] == local['recovery_target_ids']
    assert result['all_except_existing_extra_frozen'] is True
    assert result['full62_references_saved_during_development'] is False
    assert result['compact_development_only'] is True
    assert result['fresh_Adam_initialization'] is True
    assert result['same_start_for_b_c'] is True
    return dict(historical_discarded_Adam=240, arm=name,
                recorded_live_steps=120, recorded_shadow_steps=120,
                same_original_source_and_fresh_Adam_proven=True,
                all_Actor_Adam_and_gradient_events_exact=True,
                all_non_extra_parameters_exact=True, trainable_parameters=4450,
                actual_current_anchor_histories=len(anchors), recovery_histories=7,
                original322_success_retention_and_new_success_acceptance_checked=True,
                whole_Actor_Adam_rollback_or_selection_exact=True,
                development329_compact_only=True,
                absent_development120_arrays_not_reconstructed=True)


def selected_arm(C):
    eligible = []
    for name in ['b', 'c']:
        path = C.OUT / name / 'summary.json'
        if not path.exists():
            continue
        summary = C.read(path)
        if summary['state'] != 'COMPLETE':
            continue
        assert C.read(C.OUT / name / 'audit.json')['state'] == 'PASS'
        result = summary['result']
        if result['accepted']:
            eligible.append((len(result['selected_success_ids']), name, result))
    assert eligible
    # Stable fixed B-first tie break, using only accepted final proposals.
    return sorted(eligible, key=lambda v: (-v[0], v[1]))[0][1:]


def gate_audit(C, p, out, summary, rows, replays, guard):
    result = summary['result']
    name, prior = selected_arm(C)
    model = result['model']
    assert result['selected_arm'] == name and prior['development_all_safe']
    assert model == prior['model'] and sha(model['path']) == model['sha256']
    assert result['development_file'] == prior['development_file']
    development = C.read(result['development_file'])
    assert development['all_safe'] and len(development['success_ids']) == 329
    assert development['policy_hash'] == model['policy_hash']
    compact = {r['job_id']: r for r in development['rows']}
    assert len(compact) == 329 and all(not r['full'] for r in compact.values())
    fresh_ids = {j['id'] for j in p['fresh_jobs']}
    reference_jobs = [j for j in p['reload_jobs'] if j['id'] not in fresh_ids]
    rebuilt = [r for r in rows if r['role'] == 'REBUILD62']
    fresh = [r for r in rows if r['role'] == 'FRESH16']
    assert len(reference_jobs) == len(rebuilt) == 62
    assert [r['job_id'] for r in rebuilt] == [j['id'] for j in reference_jobs]
    assert len(fresh) == 16 and [r['job_id'] for r in fresh] == [j['id'] for j in p['fresh_jobs']]
    assert rows == rebuilt + fresh
    for row in rows:
        guard()
        assert row['full'] and row['mode'] == 'student'
        assert not row['teacher'] and row['seed'] is None
        assert row['loaded_model_sha256'] == model['sha256']
        assert row['policy_hash'] == model['policy_hash']
        assert row['request']['model'] == model['path']
        if row['role'] == 'REBUILD62':
            safe_student(row, model)
            same_physics(row, compact[row['job_id']])
    assert sorted(result['reference62_paths']) == sorted(r['path'] for r in rebuilt)
    for path, value in p['fresh_ledger'].items():
        assert sha(path) == value
    assert result['fresh16'] == C.E.tally(fresh)
    passed = all(r['success'] and not r['collision'] for r in fresh)
    references = {r['job_id']: r for r in rows}
    if passed:
        assert len(replays) == 78
        assert [r['job_id'] for r in replays] == [j['id'] for j in p['reload_jobs']]
        for row in replays:
            guard()
            old = references[row['job_id']]
            assert row['role'] == 'RELOAD78' and row['reference_path'] == old['path']
            assert row['request'] == old['request']
            assert row['loaded_model_sha256'] == model['sha256']
            assert row['exact'] and row['all_arrays_exact'] and not row['independent_sample']
            same_physics(row, old)
        assert result['reload78'] == dict(exact=78, all_arrays_exact=True, independent_sample=False)
    else:
        assert not replays and result['reload78'] is None
    assert result['six_gate'] == passed
    assert result['reference62_saved_in_original329'] is False
    assert result['reference62_rebuilt_native_exact'] is True
    return dict(historical_discarded_Adam=0, six_gate=passed,
                selected_arm=name, audited_complete329_predecessor_checked=True,
                native_full62_rebuilds_match_original_compact_full_array_hashes=True,
                actual_full62_policy_histories_recomputed=True,
                heldout_navigation_cases=16, independent_native_reloads=len(replays),
                reload_every_original_array_and_integration_hash_exact=True)


def eight_audit(C, p, out, summary, rows, replays, guard):
    result = summary['result']
    assert not replays
    gate = C.read(C.OUT / 'gate/summary.json')
    assert gate['state'] == 'COMPLETE' and gate['result']['six_gate'] is True
    assert C.read(C.OUT / 'gate/audit.json')['state'] == 'PASS'
    assert result['model'] == gate['result']['model']
    geometry = locked(C, p, p['eight_geometry_path'])
    assert len(geometry) == len(p['eight_jobs']) == 56
    assert [g['id'] for g in geometry] == [j['id'] for j in p['eight_jobs']]
    valid = {g['id'] for g in geometry if g['valid']}
    assert [r['job_id'] for r in rows] == [j['id'] for j in p['eight_jobs'] if j['id'] in valid]
    assert result['invalid_geometry'] == [g for g in geometry if not g['valid']]
    assert all(r['role'] == 'EIGHT_DIAGNOSTIC' and not r['full'] and not r['teacher']
               and r['seed'] is None and r['mode'] == 'student'
               and r['loaded_model_sha256'] == result['model']['sha256']
               and r['policy_hash'] == result['model']['policy_hash'] for r in rows)
    assert result['diagnostic'] == C.E.tally(rows)
    assert result['eight_started'] is True and result['eight_training'] is False
    assert result['seven_skipped'] and result['six_gate_previously_passed']
    assert result['not_eight_stable_acceptance'] is True
    return dict(historical_discarded_Adam=0, eight_frozen_diagnostic_only=True,
                registered56_geometry_inputs_verified=True,
                seven_skipped_by_user_authorization=True,
                compact_eight_evidence_only=True, no_eight_stability_claim=True)


def main(name):
    import t432_core as C
    assert name in ['data', 'b', 'c', 'gate', 'eight']
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
        def prohibited(*_args, **_kwargs):
            raise AssertionError('Independent T432 audit forbids native/exports/other optimizers')
        C.E.L.Session = prohibited
        C.Pool = prohibited
        C.E.Pool = prohibited
        C.E.N.S.P.C.B.setup = prohibited
        actual_adam = [0]
        original_adam = C.torch.optim.Adam.step
        def counted_adam(optimizer, *args, **kwargs):
            guard()
            assert name in ['b', 'c'] and actual_adam[0] < 240
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
        elif name in ['b', 'c']:
            details = arm_audit(C, p, name, out, summary, rows, replays, guard)
        elif name == 'gate':
            details = gate_audit(C, p, out, summary, rows, replays, guard)
        else:
            details = eight_audit(C, p, out, summary, rows, replays, guard)
        assert actual_adam[0] == details['historical_discarded_Adam']
        assert [r['id'] for r in commands] == list(range(1, len(commands) + 1))
        assert len(commands) == summary['commands'] == 4 + len(rows) + len(replays) <= budget['max_commands']
        assert [(r['role'], r['physics']) for r in commands[:4]] == [('SETUP', 0)] * 4
        expected = [(r['role'], r['physics'], r['pid']) for r in rows + replays]
        actual = [(r['role'], r['physics'], r['pid']) for r in commands[4:]]
        assert sorted(expected) == sorted(actual)
        for role in {entry[0] for entry in expected}:
            assert [entry for entry in actual if entry[0] == role] == [entry for entry in expected if entry[0] == role]
        assert sum(r['physics'] for r in commands) == summary['physics'] <= budget['max_physics']
        assert summary['elapsed_s'] <= budget['main_seconds']
        assert C.E.N.K.S.used(out) <= budget['max_bytes']
        guard()
        C.verify(p)
        C.save(out / 'audit.json', dict(state='PASS', stage=name, native=0,
            new_navigation_training=0, elapsed_s=time.time() - start,
            native_processes_absent=absent, manifest_members_before_audit=members,
            physics=summary['physics'], commands=summary['commands'],
            raw_commands_physics_and_termination_ledger_exact=True,
            full_actual_history_hidden_densities_rewards_checked=True,
            compact_evidence_limitation='Compact cases retain actions/counts/flags and full array hashes; absent120 arrays are not reconstructed by zero-native audit.',
            original_T422_training_audit_not_repaired=True, **details))
        C.E.N.manifest(out)
        print('T432_STAGE_AUDIT_PASS', name, flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    assert len(sys.argv) == 2
    try:
        main(sys.argv[1])
    except BaseException:
        try:
            import t432_core as C
            target = C.OUT / sys.argv[1] / 'audit_error.json'
            if not target.exists():
                C.save(target, dict(state='FAILED', error=traceback.format_exc(),
                                   epoch=time.time(), no_native=True, no_restart=True))
        finally:
            raise
