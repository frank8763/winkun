"""Stage-only, zero-optimizer fixed-feedback recovery diagnostic."""
import t429_core as C
from t429_core import *


class Runtime:
    """T427 runtime with a separate output root and no learning phase."""
    def __init__(self, p):
        self.p = p
        self.folder = OUT / 'native'
        self.folder.mkdir()
        self.limit = dict(max_physics=p['max_physics'], max_commands=p['max_commands'],
                          max_bytes=p['max_bytes'], main_seconds=p['main_seconds'])
        self.start = time.time()
        self.deadline = min(self.start + self.limit['main_seconds'], p['deadline_native'])
        self.physics = mp.get_context('spawn').RawArray('q', 5)
        self.commands = 0
        self.rows, self.replays = [], []
        self.last = 0.
        self.role = 'SETUP'
        self.pool = None
        self.lock = (ROOT / 'stage/runs/.stage_writer.lock').open('a+b')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(self.folder / 'main_started.json', dict(pid=os.getpid(), epoch=self.start,
             deadline=self.deadline, stage='native', Actor_updates=0, optimizer_calls=0))
        E.set_out(self.folder)

    def start_pool(self):
        cfg = dict(self.p, out=str(self.folder), native_stop_epoch=self.deadline,
                   max_physics=self.limit['max_physics'])
        self.pool = Pool(cfg, self.physics, self.guard)
        self.commands = 4
        for cid, row in enumerate(self.pool.ready, 1):
            self.log(cid, 'SETUP', 0, row['pid'])

    def guard(self):
        E.L.check_stop()
        assert time.time() < self.deadline - 5
        assert sum(self.physics) <= self.limit['max_physics']
        assert self.commands <= self.limit['max_commands']
        if time.time() - self.last > 5:
            self.last = time.time()
            verify(self.p)
            assert E.N.K.S.used(self.folder) <= self.limit['max_bytes']
            save(self.folder / 'status.json', dict(state=self.role, epoch=self.last,
                 pid=os.getpid(), physics=sum(self.physics), commands=self.commands,
                 rows=len(self.rows), Actor_updates=0, optimizer_calls=0))

    def log(self, cid, role, physics, pid):
        with (self.folder / 'commands.jsonl').open('a') as f:
            f.write(json.dumps(dict(id=cid, role=role, physics=physics, pid=pid)) + '\n')

    def request(self, requests, role, full=False, refs=None):
        self.role = role
        output = []
        if refs is not None:
            assert len(refs) == len(requests)
        for offset in range(0, len(requests), 4):
            req = requests[offset:offset + 4]
            self.guard()
            assert self.commands + len(req) <= self.limit['max_commands']
            assert sum(self.physics) + sum(x['job']['case']['step_cap'] for x in req) <= self.limit['max_physics']
            ids = list(range(self.commands + 1, self.commands + len(req) + 1))
            self.commands += len(req)
            values = self.pool.request(req)
            assert len(values) == len(req)
            for i, ((a, native_row), request, cid) in enumerate(zip(values, req, ids)):
                row = dict(native_row, role=role, job_id=request['job']['id'],
                           physics_steps=int(a['native_counts'].sum()),
                           loaded_model_sha256=sha(request['model']),
                           seed=request.get('seed'), request=copy.deepcopy(request))
                assert row['physics_steps'] == row['physics']
                if refs is not None:
                    old, previous = refs[offset + i]
                    assert set(a) == set(old)
                    for key in a:
                        assert E.equal(a[key], old[key]), (role, row['job_id'], key)
                    assert row['array_digest'] == previous['array_digest']
                    assert row['integration_sha256'] == previous['integration_sha256']
                    row.update(exact=True, all_arrays_exact=True, independent_sample=False,
                               reference_path=previous['path'])
                    self.replays.append(row)
                    name = 'replays.jsonl'
                else:
                    keep_full = full(request) if callable(full) else full
                    path = self.folder / 'episodes' / f'{len(self.rows):04}.npz'
                    path.parent.mkdir(exist_ok=True)
                    E.N.write_npz(path, E.N.S.packed(a) if keep_full else
                                  {k: a[k] for k in ['actions', 'native_counts', 'end_flags']},
                                  self.p['full_bytes'] if keep_full else 5 * 1024)
                    row.update(path=str(path), sha256=sha(path), full=bool(keep_full))
                    self.rows.append(row)
                    name = 'cases.jsonl'
                with (self.folder / name).open('a') as f:
                    f.write(json.dumps(row) + '\n')
                self.log(cid, role, row['physics'], row['pid'])
                output.append((a, row))
        return output

    def student(self, jobs, model, role, full=False, seeds=None, refs=None):
        return self.request([dict(model=model['path'], job=j, mode='student',
                                  seed=None if seeds is None else seeds[i])
                             for i, j in enumerate(jobs)], role, full, refs)

    def close(self, state, result, error=None):
        if self.pool:
            self.pool.close()
        save(self.folder / 'summary.json', dict(state=state, pid=os.getpid(),
             elapsed_s=time.time() - self.start, physics=sum(self.physics),
             commands=self.commands, result=result, error=error, Actor_updates=0,
             optimizer_calls=0, all_workers_stopped=True, all_sessions_closed=True,
             no_restart=True))
        save(self.folder / 'status.json', dict(state=state, epoch=time.time(),
             physics=sum(self.physics), commands=self.commands))
        E.N.manifest(self.folder)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def authenticated(row):
    assert row['full'] and sha(row['path']) == row['sha256']
    a = E.N.S.arrays(row['path'])
    assert set(a) == set(row['array_hashes'])
    assert digest(a) == row['array_digest']
    for key, value in a.items():
        assert digest(value) == row['array_hashes'][key], (row['job_id'], key)
    assert digest(a['final_native_integration']) == row['integration_sha256']
    assert int(a['native_counts'].sum()) == row['physics_steps']
    return a


def prefix_check(a, row, target):
    source_row = target['student_row']
    source = authenticated(source_row)
    first = target['first']
    assert isinstance(first, int) and 0 <= first < len(source['actions'])
    assert first < len(a['actions'])
    assert row['policy_hash'] == source_row['policy_hash']
    assert row['loaded_model_sha256'] == source_row['loaded_model_sha256']
    assert E.equal(a['ownership'][:first], np.zeros(first, np.uint8))
    assert np.all(a['ownership'][first:] == 1)
    assert set(a) == set(source) | {'feedback_progress', 'feedback_speed'}
    histories = ['history_poses', 'history_goals', 'history_times']
    constants = ['initial_native_integration', 'task_goal_xy', 'scene_sha256']
    terminal = ['final_native_integration', 'last_hidden_after']
    row_keys = []
    for key in source:
        if key in histories:
            assert E.equal(a[key][:first + 1], source[key][:first + 1]), key
        elif key in constants:
            assert E.equal(a[key], source[key]), key
        elif key not in terminal:
            assert len(source[key]) == len(source['actions']), key
            assert E.equal(a[key][:first], source[key][:first]), key
            row_keys.append(key)
    for key in ['states', 'poses', 'hidden_before']:
        assert E.equal(a[key][:first + 1], source[key][:first + 1]), ('handoff', key)
    assert np.all(a['native_counts'][:first] == 100)
    assert np.all(a['feedback_progress'][:first] == -1)
    assert np.all(a['feedback_speed'][:first] == 0)
    ref = C.R.load_reference(target['reference_path'], target['reference_sha256'])
    assert E.equal(ref['goal'], a['task_goal_xy'])
    progress = first
    elapsed = int(a['native_counts'][:first].sum())
    for i in range(first, len(a['actions'])):
        speed = float(a['states'][i, 56])
        action, progress, _ = C.R.feedback(ref, a['poses'][i], speed,
                                          elapsed * .002, progress,
                                          row['request']['gains'], 'progress')
        assert E.equal(action, a['actions'][i]), (row['job_id'], i, 'feedback_action')
        assert progress == int(a['feedback_progress'][i])
        assert E.equal(np.asarray(np.float32(speed)), np.asarray(a['feedback_speed'][i]))
        elapsed += int(a['native_counts'][i])
    return dict(id=row['job_id'], first=first, source_path=source_row['path'],
                source_sha256=source_row['sha256'], source_array_digest=source_row['array_digest'],
                reference_path=target['reference_path'], reference_sha256=target['reference_sha256'],
                prefix_exact=True, prefix_keys=sorted(row_keys), history_keys=histories,
                constant_keys=constants, handoff_state_exact=True, handoff_hidden_exact=True,
                feedback_recomputed_exact=True, speed_packet_column=56,
                native_prefix_steps=int(a['native_counts'][:first].sum()),
                no_native_state_or_clock_or_history_reset=True)


def native(rt, p):
    targets = p['recoveries']
    assert len(targets) == 5 and len({t['job_id'] for t in targets}) == 5
    assert p['jobs'] == [t['job'] for t in targets]
    model = C.initial_record(p)
    assert model['path'] == p['student_start_model']
    assert model['sha256'] == p['student_start_sha256']
    assert model['policy_hash'] == p['expected_policy_hash']
    assert sha(model['path']) == model['sha256']
    assert frozen(p, model['path']).phash() == model['policy_hash']
    requests = []
    for target in targets:
        assert target['job_id'] == target['job']['id']
        student_row, failed = target['student_row'], target['failed_mixed_row']
        assert student_row['job_id'] == failed['job_id'] == target['job_id']
        authenticated(student_row)
        authenticated(failed)
        assert not failed['success'] or failed['collision']
        assert failed['first'] == target['first']
        assert failed['request']['first'] == target['first']
        assert failed['request']['job'] == target['job']
        assert student_row['loaded_model_sha256'] == model['sha256']
        assert student_row['policy_hash'] == model['policy_hash']
        C.R.load_reference(target['reference_path'], target['reference_sha256'])
        requests.append(dict(model=model['path'], job=target['job'], mode='feedback',
                             first=target['first'], seed=None, gains=p['gains'],
                             reference_path=target['reference_path'],
                             reference_sha256=target['reference_sha256']))
    values = rt.request(requests, 'FEEDBACK_RECOVERY', True)
    # Every completed mixed rollout is replayed, including failed recoveries.
    rt.request(requests, 'FEEDBACK_EXACT', True, values)
    checks = [prefix_check(a, row, target) for (a, row), target in zip(values, targets)]
    save(rt.folder / 'prefix_checks.json', checks)
    return dict(student_source=model, recoveries=[r for a, r in values],
                total_safe=sum(bool(r['success'] and not r['collision']) for a, r in values),
                total_failed=sum(bool(not r['success'] or r['collision']) for a, r in values),
                prefix_checks=checks, exact_replays=len(values), gains=p['gains'],
                Actor_updates=0, optimizer_calls=0, not_navigation_acceptance=True,
                six_gate=False, eight_started=False, training_only_controller_diagnostic=True)


def main():
    E.N.offline()
    p = read(OUT / 'protocol.json')
    verify(p)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ban_learning()
    E.L.install_stop_handlers()
    rt, state, result, error = None, 'FAILED', {}, None
    try:
        rt = Runtime(p)
        rt.start_pool()
        result = native(rt, p)
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        if rt:
            save(rt.folder / 'failure.json', dict(error=error, role=rt.role,
                 physics=sum(rt.physics), commands=rt.commands, no_restart=True))
    finally:
        if rt:
            rt.close(state, result, error)
    assert state == 'COMPLETE', error
    print('T429_NATIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
