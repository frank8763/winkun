"""Two independent three-Adam contracts for EACH T432 B/C learner.

Each probe uses two authenticated current-policy anchors and one real
feedback recovery. Each arm uses A/B processes; all twelve updates across four probes are discarded: a fresh one-step fit,
then matching live/shadow steps from the resulting mature Adam state.
Rejection and interruption restoration are checked without further updates.
No native rollout, model export or retained learning is performed here.
Import performs no probe, data read, optimizer construction or filesystem write.
"""
from pathlib import Path
import copy
import os
import platform
import sys
import time
import t432_learning as L


def _runtime(torch, np):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=str(np.__version__), default_dtype=str(torch.get_default_dtype()),
        intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        cpu_autocast=torch.is_autocast_enabled('cpu'), grad_enabled=torch.is_grad_enabled(),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
        MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
        OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'))


def fixture_sources(C, p):
    """Select fixed current-policy rows; never construct or run an Actor."""
    rows = p['current_anchor_rows']
    assert len(rows) == 59 and len({r['job_id'] for r in rows}) == 59
    assert {r['policy_hash'] for r in rows} == {p['expected_policy_hash']}
    old_ids = {job['id'] for job in p['old8_jobs']}
    six_ids = {job['id'] for job in p['main_jobs']}
    assert sum(r['job_id'] in old_ids for r in rows) == 8
    assert sum(r['job_id'] in six_ids for r in rows) == 51
    anchors = [next(r for r in rows if r['job_id'] in old_ids),
               next(r for r in rows if r['job_id'] in six_ids)]
    target_file = p['verified_target_file']
    assert C.sha(target_file) == p['inputs'][target_file]
    target_audit = p['verified_target_audit']
    assert C.sha(target_audit) == p['inputs'][target_audit]
    assert C.read(target_audit)['state'] == 'PASS'
    targets = C.read(target_file)
    assert len(targets) == 7
    entry = next(t for t in targets if t['request']['mode'] == 'feedback')
    row = entry['row']
    assert row['policy_hash'] == p['expected_policy_hash']
    assert row['success'] and not row['collision'] and entry['exact']
    target = dict(path=entry['path'], sha256=entry['sha256'],
                  array_digest=row['array_digest'], policy_hash=row['policy_hash'],
                  first_feedback_or_teacher_hold=entry['first'], job_id=entry['id'],
                  source_audit_path=target_audit, source_audit_sha256=C.sha(target_audit),
                  on_policy_for_PPO=False)
    return anchors, target


def _parameter_mapping(q, opt):
    names = {id(value): name for name, value in q.named_parameters()}
    ordered = [names[id(value)] for group in opt.param_groups for value in group['params']]
    variables = L.params(q)
    assert ordered == [names[id(value)] for value in variables]
    assert len(ordered) == len(set(ordered)) == 4
    assert set(ordered) == {'extra.0.weight', 'extra.0.bias', 'extra.2.weight', 'extra.2.bias'}
    assert sum(v.numel() for v in variables) == 4450
    mapping = [dict(index=k, name=names[id(value)], shape=list(value.shape), dtype=str(value.dtype))
               for k, value in enumerate(variables)]
    return ordered, mapping


def _write(C, path, value):
    assert path.parent == C.OUT
    assert path.name in [f'clone_{arm}_{label}.json' for arm in ['b', 'c'] for label in ['A', 'B']] + ['compare_b.json', 'compare_c.json']
    assert not path.exists(), ('Do not repeat completed clone receipt', path)
    C.save(path, value)


def probe(arm, label):
    import t432_core as C
    assert arm in ['b', 'c'] and label in ['A', 'B']
    E, torch, np = C.E, C.torch, C.np
    p = C.read(C.OUT / 'protocol.json')
    C.verify(p)
    p = dict(p, arm=arm, phase_arm=arm)
    assert not (C.OUT / ('clone_' + arm + '_' + label + '.json')).exists()
    started = time.time()
    deadline = min(started + 180, p['deadline_all'], p['deadline_learning'])
    E.N.offline()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def guard():
        assert time.time() < deadline - 5

    def forbidden(*args, **kwargs):
        raise AssertionError('T432 clone contract forbids native simulation and model export')

    E.L.Session = forbidden
    E.Pool = forbidden
    C.Pool = forbidden
    E.N.S.P.C.B.setup = forbidden
    E.export = forbidden
    E.N.write_npz = forbidden
    L.export_model = forbidden
    L.PREVIOUS.export_model = forbidden
    for optimizer in [torch.optim.AdamW, torch.optim.SGD]:
        optimizer.step = forbidden

    assert C.sha(p['student_start_model']) == p['student_start_sha256']
    q = C.frozen(p)
    initial_policy_hash = q.phash()
    assert initial_policy_hash == p['expected_policy_hash']
    opt = L.make_optimizer(q, p)
    assert not opt.state, 'Checkpoint optimizer must not be inherited'
    ordered, mapping = _parameter_mapping(q, opt)
    fixed = L.fixedhash(q)
    before_audit = C.digest(q.state_dict())
    anchor_rows, target_row = fixture_sources(C, p)
    anchors, sources = [], []
    for row in anchor_rows:
        guard()
        assert row['full'] and row['success'] and not row['collision'] and not row['timeout']
        assert row['loaded_model_sha256'] == p['student_start_sha256']
        assert C.sha(row['path']) == row['sha256']
        arrays = E.N.S.arrays(row['path'])
        assert C.digest(arrays) == row['array_digest']
        E.audit_array(q, arrays, p, 0, None)
        anchors.append((arrays, row))
        sources.append(dict(kind='authenticated_current_policy_success_anchor',
            path=row['path'], sha256=row['sha256'], id=row['job_id'],
            policy_hash=row['policy_hash'], array_digest=row['array_digest']))
    guard()
    assert C.sha(target_row['path']) == target_row['sha256']
    assert C.sha(target_row['source_audit_path']) == target_row['source_audit_sha256']
    assert C.read(target_row['source_audit_path'])['state'] == 'PASS'
    arrays = E.N.S.arrays(target_row['path'])
    assert C.digest(arrays) == target_row['array_digest']
    first = target_row['first_feedback_or_teacher_hold']
    E.audit_array(q, arrays, p, first, None)
    targets = [dict(arrays=arrays, first=first, id=target_row['job_id'])]
    sources.append(dict(kind='authenticated_actual_current_state_feedback_recovery',
        path=target_row['path'], sha256=target_row['sha256'], id=target_row['job_id'],
        policy_hash=target_row['policy_hash'], array_digest=target_row['array_digest'],
        first=first, source_audit_path=target_row['source_audit_path'],
        source_audit_sha256=target_row['source_audit_sha256'], not_onpolicy=True))
    assert C.digest(q.state_dict()) == before_audit and q.phash() == initial_policy_hash
    q.reset_noise()
    pp = copy.deepcopy(p)
    pp['preflight_clone_probe'] = True
    groups = L.groups(q, pp, anchors, targets)
    initial = E.L.statehash(q, opt)
    runtime = _runtime(torch, np)
    events = []
    calls = 0
    permitted = [opt]
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        nonlocal calls
        guard()
        assert any(optimizer is allowed for allowed in permitted)
        assert calls < 3, 'At most three discarded Adam calls per probe'
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        return result

    def event(which, model, optimizer):
        def emit(where, k):
            guard()
            assert where in ['BEFORE', 'AFTER'] and k == 0
            gradient = {name: value.grad for name, value in model.named_parameters() if name in ordered}
            assert all(value is not None for value in gradient.values())
            events.append(dict(which=which, where=where, k=k,
                hashes=E.L.statehash(model, optimizer), gradient_sha256=C.digest(gradient),
                gradient_dtypes={name: str(value.dtype) for name, value in gradient.items()}))
        return emit

    torch.optim.Adam.step = step
    try:
        fresh_fit = L.fit(q, opt, groups, pp, guard, event('fresh', q, opt), steps=1)
        assert calls == 1
        mature_before = copy.deepcopy(E.L.state(q, opt))
        mature_before_hashes = E.L.statehash(q, opt)
        assert initial['learned'] != mature_before_hashes['learned']
        assert opt.state and all(float(state['step']) == 1 for state in opt.state.values())
        shadow = C.frozen(p)
        shadow_opt = L.make_optimizer(shadow, p)
        assert not shadow_opt.state
        assert _parameter_mapping(shadow, shadow_opt) == (ordered, mapping)
        E.L.load(shadow, shadow_opt, mature_before)
        permitted.append(shadow_opt)
        assert E.L.statehash(shadow, shadow_opt) == mature_before_hashes
        live_fit = L.fit(q, opt, groups, pp, guard, event('mature_live', q, opt), steps=1)
        mature_after = copy.deepcopy(E.L.state(q, opt))
        mature_after_hashes = E.L.statehash(q, opt)
        shadow_fit = L.fit(shadow, shadow_opt, groups, pp, guard,
                           event('mature_shadow', shadow, shadow_opt), steps=1)
        assert calls == 3 and live_fit == shadow_fit
        assert mature_after_hashes == E.L.statehash(shadow, shadow_opt)
        assert mature_before_hashes != mature_after_hashes
        live_events = [{k: v for k, v in row.items() if k != 'which'}
                       for row in events if row['which'] == 'mature_live']
        shadow_events = [{k: v for k, v in row.items() if k != 'which'}
                         for row in events if row['which'] == 'mature_shadow']
        assert len(events) == 6 and len(live_events) == len(shadow_events) == 2
        assert live_events == shadow_events
        assert L.fixedhash(q) == L.fixedhash(shadow) == fixed

        # Reject the already computed mature proposal; no fourth fit occurs.
        E.L.load(q, opt, mature_before)
        rejection_restored = E.L.statehash(q, opt)
        assert rejection_restored == mature_before_hashes and L.fixedhash(q) == fixed

        class SimulatedInterruption(Exception):
            pass

        try:
            # Reuse the same computed proposal to simulate an interruption
            # after an update, rather than spending another optimizer call.
            E.L.load(q, opt, mature_after)
            assert E.L.statehash(q, opt) == mature_after_hashes
            raise SimulatedInterruption('T432 discarded transaction probe')
        except SimulatedInterruption:
            E.L.load(q, opt, mature_before)
        interruption_restored = E.L.statehash(q, opt)
        assert interruption_restored == mature_before_hashes and L.fixedhash(q) == fixed
        assert calls == 3
    finally:
        torch.optim.Adam.step = original_step

    contract = dict(protocol_sha256=C.sha(C.OUT / 'protocol.json'),
        source_files={str(f): C.sha(f) for f in [C.HERE / 't432_contract.py',
            C.HERE / 't432_learning.py', C.HERE / 't432_core.py',
            C.ROOT / 'stage/experiments/t431_actual_recovery/t431_learning.py']},
        starting_model_sha256=p['student_start_sha256'], starting_policy_hash=initial_policy_hash,
        arm=arm, trainable_parameters=4450,
        recovery_targets_sha256=C.sha(p['verified_target_file']),
        fixture_sources=sources, fixtures_are_existing_current_policy_evidence=True,
        no_new_native_successes_claimed=True, full_student_mean_history_density_replay_passed=True,
        ordered_parameter_mapping=mapping, optimizer_defaults=opt.defaults,
        param_groups=opt.state_dict()['param_groups'], runtime_before=runtime,
        runtime_after=_runtime(torch, np), before_hashes=initial,
        mature_before_hashes=mature_before_hashes, mature_after_hashes=mature_after_hashes,
        shadow_after_hashes=E.L.statehash(shadow, shadow_opt),
        rejection_restored_hashes=rejection_restored,
        interruption_restored_hashes=interruption_restored,
        mature_Actor_Adam_rejection_and_interruption_restore_exact=True,
        clipped_gradients_and_states=events, fresh_fit=fresh_fit,
        mature_live_fit=live_fit, mature_shadow_fit=shadow_fit,
        mature_live_shadow_gradients_and_states_exact=True,
        frozen_original_before=fixed, frozen_original_after=L.fixedhash(q),
        native=0, new_navigation_training=0, retained_Actor=0,
        discarded_fresh_Adam=1, discarded_mature_live_Adam=1,
        discarded_mature_shadow_Adam=1, discarded_Adam_total=calls,
        no_model_export=True, old_T422_training_audit_not_repaired=True)
    _write(C, C.OUT / ('clone_' + arm + '_' + label + '.json'),
           dict(state='THREE_DISCARDED_ADAM_CONTRACT_COMPLETE', arm=arm, label=label,
                pid=os.getpid(), elapsed_s=time.time() - started, contract=contract))
    print('T432_CLONE_WRITTEN', arm, label, 'discarded_Adam', calls, flush=True)


def compare(arm):
    import t432_core as C
    assert arm in ['b', 'c']
    a = C.read(C.OUT / ('clone_' + arm + '_A.json'))
    b = C.read(C.OUT / ('clone_' + arm + '_B.json'))
    assert a['state'] == b['state'] == 'THREE_DISCARDED_ADAM_CONTRACT_COMPLETE'
    assert a['arm'] == b['arm'] == arm
    assert a['label'] == 'A' and b['label'] == 'B' and a['pid'] != b['pid']
    assert a['contract'] == b['contract'], 'Independent-process numerical contracts differ'
    contract = a['contract']
    assert contract['arm'] == arm and contract['trainable_parameters'] == 4450
    assert contract['discarded_Adam_total'] == 3
    assert contract['native'] == contract['new_navigation_training'] == contract['retained_Actor'] == 0
    assert contract['mature_Actor_Adam_rejection_and_interruption_restore_exact']
    assert contract['mature_live_shadow_gradients_and_states_exact']
    _write(C, C.OUT / ('compare_' + arm + '.json'), dict(state='PASS_NEW_THREE_STEP_CROSS_PROCESS_CONTRACT',
        arm=arm, trainable_parameters=4450,
        processes_distinct=True, full_numeric_contract_exact=True,
        Actor_Adam_parameter_order_gradients_runtime_exact=True,
        mature_Actor_Adam_rejection_interruption_restore_exact=True,
        full_student_mean_history_density_replay_passed=True,
        discarded_fresh_Adam=2, discarded_mature_live_Adam=2,
        discarded_mature_shadow_Adam=2, discarded_Adam_total=6,
        native=0, new_navigation_training=0, retained_Actor=0,
        existing_fixtures_not_new_native_successes=True, old_T422_audit_remains_FAIL=True,
        clone_A_sha256=C.sha(C.OUT / ('clone_' + arm + '_A.json')),
        clone_B_sha256=C.sha(C.OUT / ('clone_' + arm + '_B.json'))))
    print('T432_NEW_THREE_STEP_CLONE_CONTRACT_PASS', arm, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'compare':
        compare(sys.argv[2])
    else:
        assert len(sys.argv) == 4 and sys.argv[1] == 'probe'
        probe(sys.argv[2], sys.argv[3])
