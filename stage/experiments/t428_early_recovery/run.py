"""Stage-only, zero-optimizer clock-aligned early recovery diagnostic."""
import t428_core as C
from t428_core import *


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


def assert_historical(a, row, previous):
    """The full historical hashes prove the rebuilt execution, without a new replay."""
    assert set(a) == set(previous['array_hashes'])
    assert digest(a) == previous['array_digest']
    for key, value in a.items():
        assert digest(value) == previous['array_hashes'][key], (previous['job_id'], key)
    assert digest(a['final_native_integration']) == previous['integration_sha256']
    for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics_steps',
                'success', 'collision', 'timeout', 'policy_hash', 'loaded_model_sha256']:
        assert row[key] == previous[key], (previous['job_id'], key)


def trigger_for(job_id, student, teacher):
    """Compare only decision starts at the same integer native physics clock."""
    clocks = []
    for a in [student, teacher]:
        counts = np.asarray(a['native_counts'])
        assert counts.ndim == 1 and len(counts) and np.issubdtype(counts.dtype, np.integer)
        assert np.all((counts > 0) & (counts <= 100))
        assert np.all(counts[:-1] == 100), 'Only the terminal hold may be partial'
        assert len(a['poses']) == len(counts)
        clocks.append(np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))[:-1])
    n = min(len(clocks[0]), len(clocks[1]))
    assert np.array_equal(clocks[0][:n], clocks[1][:n])
    s, t = np.asarray(student['poses'])[:n], np.asarray(teacher['poses'])[:n]
    assert s.shape == t.shape == (n, 3) and np.all(np.isfinite(s)) and np.all(np.isfinite(t))
    distance = np.linalg.norm(s[:, :2] - t[:, :2], axis=1)
    yaw = np.abs((s[:, 2] - t[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    over = (distance > .1) | (yaw > np.deg2rad(5.))
    hits = np.flatnonzero(over[:-2] & over[1:-1] & over[2:]) if n >= 3 else []
    first_run = int(hits[0]) if len(hits) else None
    first = max(0, first_run - 1) if first_run is not None else None
    return dict(id=job_id, status='TRIGGER' if first is not None else 'NO_TRIGGER',
                first=first, first_run_index=first_run, matched_holds=n,
                aligned_native_start_counts=clocks[0][:n].astype(int).tolist(),
                distance_m=distance.tolist(), yaw_abs_deg=np.rad2deg(yaw).tolist(),
                threshold_exceeded=over.astype(bool).tolist(), distance_threshold_m=.1,
                yaw_threshold_deg=5., consecutive_holds=3,
                trigger_native_start_count=None if first_run is None else int(clocks[0][first_run]),
                handoff_native_start_count=None if first is None else int(clocks[0][first]),
                training_privileged_offline_diagnostic=True, runtime_future_access=False)


def native(rt, p):
    previous = p['failed_rows']
    assert len(previous) == 7 and len({r['job_id'] for r in previous}) == 7
    model = C.initial_record(p)
    assert model['path'] == p['student_start_model']
    assert model['sha256'] == p['student_start_sha256']
    assert model['policy_hash'] == p['expected_policy_hash']
    assert sha(model['path']) == model['sha256']
    assert frozen(p, model['path']).phash() == model['policy_hash']
    students = {}
    compact = []
    for row in previous:
        assert row['request']['model'] == model['path']
        assert row['loaded_model_sha256'] == model['sha256'] and row['policy_hash'] == model['policy_hash']
        assert not row['success'] or row['collision']
        if row['full']:
            assert sha(row['path']) == row['sha256']
            a = E.N.S.arrays(row['path'])
            assert_historical(a, row, row)
            students[row['job_id']] = (a, row)
        else:
            compact.append(row)
    assert len(students) == 3 and len(compact) == 4
    rebuilt = rt.student([r['request']['job'] for r in compact], model, 'STUDENT_REBUILD', True)
    for (a, row), old in zip(rebuilt, compact):
        assert_historical(a, row, old)
        students[row['job_id']] = (a, row)
    baseline_requests = []
    for old in previous:
        job = old['request']['job']
        request = dict(model=model['path'], job=job, mode='actor_teacher' if job.get('wide_course') else 'safe_reset', first=0, seed=None)
        if request['mode'] == 'actor_teacher':
            request.update(teacher_model=p['teacher_A_model'], teacher_sha256=p['teacher_A_sha256'])
        baseline_requests.append(request)
    baselines = rt.request(baseline_requests, 'TEACHER_BASELINE', True)
    rt.request(baseline_requests, 'TEACHER_BASELINE_EXACT', True, baselines)
    diagnostics, recovery_requests, mixed_ids = [], [], []
    for old, (teacher, baseline), request in zip(previous, baselines, baseline_requests):
        a, source_row = students[old['job_id']]
        trigger = trigger_for(old['job_id'], a, teacher)
        item = dict(id=old['job_id'], student_row_path=source_row['path'], student_row=source_row,
                    baseline_row=baseline, trigger=trigger, recovery_row=None)
        if not baseline['success'] or baseline['collision']:
            item['recovery_skipped_reason'] = 'TEACHER_BASELINE_NOT_ACTUAL_SAFE_GOAL'
        elif trigger['first'] is None:
            item['recovery_skipped_reason'] = 'NO_TRIGGER'
        else:
            recovery_requests.append(dict(request, first=trigger['first']))
            mixed_ids.append(old['job_id'])
        diagnostics.append(item)
    save(rt.folder / 'triggers.json', [d['trigger'] for d in diagnostics])
    save(rt.folder / 'diagnostics.json', diagnostics)
    recoveries = rt.request(recovery_requests, 'RECOVERY', True)
    rt.request(recovery_requests, 'RECOVERY_EXACT', True, recoveries)
    by_id = {d['id']: d for d in diagnostics}
    for job_id, (a, row) in zip(mixed_ids, recoveries):
        by_id[job_id]['recovery_row'] = row
    result = dict(student_source=model, baseline_rows=[r for a, r in baselines],
                  diagnostics=diagnostics,
                  actual_safe_recoveries=sum(bool(r['success'] and not r['collision']) for a, r in recoveries),
                  actual_failed_recoveries=sum(bool(not r['success'] or r['collision']) for a, r in recoveries),
                  recovery_rows=[r for a, r in recoveries],
                  student_rebuilds=4, reused_full_student_rows=3,
                  baselines_replayed_exact=len(baselines), recoveries_replayed_exact=len(recoveries),
                  not_navigation_acceptance=True, Actor_updates=0, optimizer_calls=0,
                  six_gate=False, eight_started=False)
    save(rt.folder / 'diagnostics.json', diagnostics)
    return result


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
    print('T428_NATIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
