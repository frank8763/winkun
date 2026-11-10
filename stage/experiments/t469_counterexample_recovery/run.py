"""Frozen A325 counterexamples and fixed PPO327 recovery; zero optimizers.

The two autonomous policies repeat already consumed development tasks.  Four
recovery queries are diagnostic off-policy trajectories, never new on-policy
RL data. Any later finite preservation use requires a separate registration.
Only preflight, main and audit are exposed by the CLI; registration is separate.
"""
from pathlib import Path
import importlib.util, json, os, shutil, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t469_counterexample_recovery'
DOC = ROOT/'stage/docs/overnight_20261110'
spec = importlib.util.spec_from_file_location(
    'stage469_pinned468_runtime', ROOT/'stage/experiments/t468_offline_recovery_night/night_runtime.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
N = A.N
J, H, M, P, E, R, torch, np = A.J, A.H, A.M, A.P, A.E, A.R, A.torch, A.np
read, save, sha, digest, frozen = A.read, A.save, A.sha, A.digest, N.frozen
JOB_IDS = ['dev_task6_03', 'dev_task6_15',
           'wide117_80_dev6_LEFT_15', 'wide117_80_dev6_LEFT_17']
TEACHER_KIND = 'T463_PPO327_actual_history_carry'
MIN_DECISION, CONSECUTIVE, STEER_THRESHOLD, FALLBACK = 20, 3, .02, 20


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert p['out'] == str(OUT)
    assert [j['id'] for j in p['diagnostic_jobs']] == JOB_IDS
    assert set(p['diagnostic_reference_student']) == set(JOB_IDS)
    assert set(p['diagnostic_reference_teacher']) == set(JOB_IDS)
    assert p['max_bytes'] == 18*2**20 and p['max_physics'] == 1000000
    assert p['max_commands'] == 80 and p['preflight_commands'] == 24
    assert p['full_bytes'] == 448*1024 and p['compact_bytes'] == 4096
    assert p['main_seconds'] == 900 and p['learning_seconds'] == 1
    assert p['learning_initial']['policy_hash'] != p['teacher_model']['policy_hash']
    assert M.used(OUT) <= p['max_bytes']
    assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f
    for name in ['learning_initial', 'teacher_model']:
        assert sha(p[name]['path']) == p[name]['sha256']


def teacher(p, job):
    assert job['id'] in JOB_IDS
    q = frozen(p, p['teacher_model']['path']).eval().requires_grad_(False)
    assert q.phash() == p['teacher_model']['policy_hash']
    return q, TEACHER_KIND, sha(p['teacher_model']['path'])


class _TeacherPipe:
    """Correct the old worker's kind-specific descriptive carry flag only."""
    def __init__(self, pipe): self.pipe = pipe
    def recv(self): return self.pipe.recv()
    def close(self): return self.pipe.close()
    def send(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            row = value[1]
            if row.get('teacher_kind') == TEACHER_KIND:
                row['teacher_hidden_burns_actual_history_from_h0'] = True
                row['teacher_carry_flag_adapted_for_new_kind'] = True
        return self.pipe.send(value)


def worker(pipe, index, p, counts):
    # All control, native integration and student hidden updates remain the
    # pinned T468 worker.  Its non-safe teacher_command path calls q.step.
    A.teacher = teacher
    A.worker(_TeacherPipe(pipe), index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                child = ctx.Process(target=worker, args=(b, i, p, counts))
                child.start(); b.close()
                self.children.append(child); self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(v['state'] == 'READY' for v in self.ready)
            path = Path(p['out'])/'worker_generations.json'
            generations = read(path) if path.exists() else []
            generations.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(path, generations)
        except BaseException:
            self.close(); raise


def setup_runtime():
    A.teacher = teacher
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def readonly_cpu():
    E.N.offline(); R.C.ban_learning()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def preflight():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime(); readonly_cpu()
    # Only construct frozen CPU Actors.  No optimizer or training constructor.
    for name in ['learning_initial', 'teacher_model']:
        q = frozen(p, p[name]['path']).eval().requires_grad_(False)
        assert q.phash() == p[name]['policy_hash']
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'; folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=min(time.time()+180, p['absolute_native_stop']), max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['learning_initial']['path'], job=j, seed=None, cancel_probe=True) for j in p['old8_jobs']])
        assert len(replies) == 8 and all(v['state'] == 'CANCEL_PASS' for v in replies)
        assert sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24, optimizer=0,
        Actor_Adam=0, Value_Adam=0, frozen_models_cpu_hash_exact=True,
        eight_workers_cancelled_before_first_native=True, all_workers_stopped=True,
        historical_reproduction_and_recovery_NOT_executed_yet=True))
    print('T469_PREFLIGHT_PASS', flush=True)


def original_exact(values, reference, model):
    for a, row in values:
        old = reference[row['job_id']]
        assert row['policy_hash'] == old['policy_hash'] == model['policy_hash']
        assert row['loaded_model_sha256'] == model['sha256']
        for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics',
                    'success', 'collision', 'timeout', 'administrative_censor']:
            assert row[key] == old[key], (row['job_id'], key)
        assert row['full'] and not row['teacher'] and not row['administrative_censor']


def _regions(error):
    result = {}
    for name, part in [('prefix_first21', slice(0, 21)), ('suffix21onward', slice(21, None)), ('all', slice(None))]:
        e = error[part]
        result[name] = dict(decisions=len(e),
            RMSE_per_axis=np.sqrt((e**2).mean(0)).tolist() if len(e) else None,
            max_abs_per_axis=e.max(0).tolist() if len(e) else None,
            mean_abs_per_axis=e.mean(0).tolist() if len(e) else None,
            steering_gt02_count=int((e[:, 1] > .02).sum()),
            steering_gt05_count=int((e[:, 1] > .05).sum()))
    return result


def label_statistics(p, student_rows):
    """New teacher instance/h0 per actual failed student history, CPU only."""
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert [r['job_id'] for r in student_rows] == JOB_IDS
    stats, raw = [], {}
    with torch.no_grad():
        for row in student_rows:
            a = A.arrays(row)
            q, kind, hsh = teacher(p, row['request']['job'])
            hidden, labels = None, []
            for x in a['states']:
                action, hidden = A.teacher_command(q, kind, x, hidden)
                labels.append(action[0].numpy().copy())
            labels = np.asarray(labels, np.float32)
            assert labels.shape == a['actions'].shape and np.isfinite(labels).all()
            error = np.abs(a['actions'].astype(np.float64)-labels.astype(np.float64))
            detected = None
            for i in range(MIN_DECISION+CONSECUTIVE-1, len(error)):
                if np.all(error[i-CONSECUTIVE+1:i+1, 1] > STEER_THRESHOLD):
                    detected = i; break
            first = FALLBACK if detected is None else detected
            key = row['job_id']
            raw[key+'_teacher_actions'] = labels
            raw[key+'_action_abs_error'] = error.astype(np.float32)
            raw[key+'_source_states_sha256'] = np.frombuffer(bytes.fromhex(row['array_hashes']['states']), np.uint8).copy()
            stats.append(dict(job_id=key, source_path=row['path'], source_sha256=row['sha256'],
                source_array_digest=row['array_digest'], actual_student_success=row['success'],
                actual_student_collision=row['collision'], teacher=kind, teacher_source_SHA256=hsh,
                teacher_labels_on_actual_student_history_from_h0=True, regions=_regions(error),
                onset_candidate_detection_decision=detected,
                onset_candidate_first_streak_decision=None if detected is None else detected-CONSECUTIVE+1,
                onset_candidate_time_s=None if detected is None else detected*.2,
                teacher_first=first, fallback=detected is None,
                onset_rule=dict(min_decision=MIN_DECISION, consecutive=CONSECUTIVE,
                    abs_steering_error_threshold=STEER_THRESHOLD, fallback=FALLBACK),
                statistic_NOT_causal_or_safety_proof=True))
    return stats, raw


def prefix_check(query, source):
    first = query['teacher_first']
    a, b = A.arrays(query), A.arrays(source)
    if len(a['states']) > first and len(b['states']) > first:
        A.prefix_exact(query, source, first)
        return dict(job_id=query['job_id'], teacher_first=first, executed_handover=True, prefix_exact=True)
    # A student may terminate before the fixed fallback.  Preserve this genuine
    # unsuccessful query, compare its entire available prefix, and never retry.
    assert len(a['states']) == len(b['states']) <= first
    n = len(a['states'])
    for key in ['states', 'poses', 'history_poses', 'history_goals', 'history_times', 'hidden_before']:
        assert np.array_equal(a[key], b[key]), key
    for key in ['actions', 'native_counts', 'next_states', 'end_flags', 'final_native_integration', 'initial_native_integration']:
        assert np.array_equal(a[key], b[key]), key
    assert not a['ownership'].any()
    return dict(job_id=query['job_id'], teacher_first=first, decisions=n,
                executed_handover=False, prefix_exact=True)


def main():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime(); readonly_cpu(); E.L.install_stop_handlers(); rt = M.Runtime(p)
    status, error = 'FAILED', None
    students, baselines, queries, statistics, coverage = [], [], [], [], None
    try:
        rt.pool = Pool(dict(p, native_stop_epoch=min(rt.deadline, p['absolute_native_stop'])), rt.counts, rt.guard)
        values = rt.request(p['diagnostic_jobs'], p['learning_initial'], 'STUDENT_FAILED4', full=True)
        original_exact(values, p['diagnostic_reference_student'], p['learning_initial'])
        students = [r for a, r in values]; assert all(not r['success'] for r in students)
        rt.request(p['diagnostic_jobs'], p['learning_initial'], 'STUDENT_FAILED4_EXACT', full=True, references=students)
        values = rt.request(p['diagnostic_jobs'], p['teacher_model'], 'BASELINE_SUCCESS4', full=True)
        original_exact(values, p['diagnostic_reference_teacher'], p['teacher_model'])
        baselines = [r for a, r in values]; assert all(r['success'] and not r['collision'] for r in baselines)
        rt.request(p['diagnostic_jobs'], p['teacher_model'], 'BASELINE_SUCCESS4_EXACT', full=True, references=baselines)
        statistics, _ = label_statistics(p, students)
        jobs = [dict(j, teacher_first=s['teacher_first']) for j, s in zip(p['diagnostic_jobs'], statistics)]
        values = rt.request(jobs, p['learning_initial'], 'RECOVERY_QUERY4', full=True, query=True)
        queries = [r for a, r in values]
        rt.request(jobs, p['learning_initial'], 'RECOVERY_QUERY4_EXACT', full=True, references=queries, query=True)
        prefixes = [prefix_check(q, s) for q, s in zip(queries, students)]
        coverage = dict(all_safe=all(r['success'] and not r['collision'] for r in queries),
            teacher_results=R.M.tally(queries), selection=statistics, prefixes=prefixes,
            teacher_new_queries=4, repeated_autonomous_policy_jobs=8,
            no_same_policy_historical_repeat_is_independent=True)
        save(OUT/'coverage_gate.json', coverage)
        save(OUT/'support_rows.json', dict(student_rows=students, baseline_rows=baselines,
            recovery_rows=queries, query_rows=queries, label_statistics=statistics))
        status = 'COMPLETE'  # Scientific recovery failure is a complete diagnostic.
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, raw_preserved=True, no_restart=True, optimizer=0))
    finally:
        if rt.pool: rt.pool.close(); rt.commands += 8
        save(OUT/'summary.json', dict(state=status, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            source_rows=students+baselines, student_rows=students, baseline_rows=baselines,
            query_rows=queries, recovery_rows=queries, coverage=coverage,
            all_safe=None if coverage is None else coverage['all_safe'], label_statistics=statistics,
            autonomous_historical_repeats_not_independent=True,
            optimizer=0, Actor_Adam=0, Value_Adam=0, new_PPO=0, new_Value=0,
            completed_paired_BC=0, candidate=None, proposal=None,
            teacher_executed_native_queries=len(queries), full_329=None, fresh16=None,
            reload78=None, six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=status, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN); rt.lock.close()
    assert status == 'COMPLETE', error
    print('T469_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json'); verify(p, True); summary = read(OUT/'summary.json')
    assert summary['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    setup_runtime(); readonly_cpu()
    generations = read(OUT/'worker_generations.json')
    assert len(generations) == 1
    ids = [summary['pid']]+[pid for g in generations for pid in g['workers']]
    preflight_generations = read(OUT/'preflight_native/worker_generations.json')
    ids += [pid for g in preflight_generations for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert len(rows) == len(replays) == 12
    assert summary['physics'] == sum(r['physics'] for r in rows+replays) <= p['max_physics']
    assert summary['commands'] == p['preflight_commands']+16+len(rows)+len(replays) == 64 <= p['max_commands']
    groups = {role: [r for r in rows if r['role'] == role]
              for role in ['STUDENT_FAILED4', 'BASELINE_SUCCESS4', 'RECOVERY_QUERY4']}
    for group in groups.values(): assert [r['job_id'] for r in group] == JOB_IDS
    students, baselines, queries = (groups[k] for k in ['STUDENT_FAILED4', 'BASELINE_SUCCESS4', 'RECOVERY_QUERY4'])
    assert students == summary['student_rows'] and baselines == summary['baseline_rows']
    assert queries == summary['query_rows'] == summary['recovery_rows']
    assert students+baselines == summary['source_rows']
    assert all(not r['success'] and not r['teacher'] for r in students)
    assert all(r['success'] and not r['collision'] and not r['teacher'] for r in baselines)
    A.teacher = teacher
    for row in rows:
        model = p['teacher_model'] if row['role'] == 'BASELINE_SUCCESS4' else p['learning_initial']
        assert row['loaded_model_sha256'] == model['sha256'] and row['policy_hash'] == model['policy_hash']
        assert row['request']['seed'] is None and row['seed'] is None
        a = A.arrays(row)
        assert a['states'].shape[1] == 120 and np.array_equal(a['states'][1:], a['next_states'][:-1])
        assert int(a['native_counts'].sum()) == row['physics'] and (a['native_counts'] > 0).all()
        assert not a['stochastic'].any() and not row['administrative_censor']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert [bool(v) for v in a['end_flags'][-1]] == [row['collision'], row['success'], row['timeout']]
        assert digest(a['final_native_integration']) == row['integration_sha256']
        for value in a.values():
            if np.issubdtype(value.dtype, np.number): assert np.isfinite(value).all()
        # A.auditrow supplies N.frozen to M.audit_full_history and restores the
        # original global afterward; it also reconstructs teacher labels from h0.
        A.auditrow(p, row)
        if row['teacher']:
            assert row['teacher_kind'] == TEACHER_KIND
            assert row['teacher_source_SHA256'] == p['teacher_model']['sha256']
            assert row['teacher_hidden_burns_actual_history_from_h0']
        else:
            assert not a['ownership'].any()
            ref = p['diagnostic_reference_teacher' if row['role'] == 'BASELINE_SUCCESS4' else 'diagnostic_reference_student'][row['job_id']]
            original_exact([(a, row)], {row['job_id']: ref}, model)
    bypath = {r['path']: r for r in rows}
    for row in replays:
        original = bypath[row['reference_path']]
        assert row['role'] == original['role']+'_EXACT'
        assert row['job_id'] == original['job_id'] and row['request'] == original['request']
        for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics',
                    'policy_hash', 'loaded_model_sha256', 'teacher', 'teacher_first',
                    'teacher_kind', 'teacher_source_SHA256', 'success', 'collision', 'timeout']:
            assert row[key] == original[key], key
        assert row['exact'] and row['independent_new_sample'] is False
    prefixes = [prefix_check(q, s) for q, s in zip(queries, students)]
    statistics, arrays = label_statistics(p, students)
    again, other = label_statistics(p, students)
    assert statistics == again == summary['label_statistics']
    assert arrays.keys() == other.keys() and all(np.array_equal(arrays[k], other[k]) for k in arrays)
    blob = E.N.R.blob(arrays)
    assert blob == E.N.R.blob(other) and len(blob) <= 512*1024
    path = OUT/'offline_teacher_labels.npz'; assert not path.exists(); path.write_bytes(blob)
    support = read(OUT/'support_rows.json')
    assert support == dict(student_rows=students, baseline_rows=baselines, recovery_rows=queries,
                           query_rows=queries, label_statistics=statistics)
    coverage = read(OUT/'coverage_gate.json'); assert coverage == summary['coverage']
    assert coverage['selection'] == statistics and coverage['prefixes'] == prefixes
    assert [r['teacher_first'] for r in queries] == [s['teacher_first'] for s in statistics]
    assert coverage['all_safe'] == summary['all_safe'] == all(r['success'] and not r['collision'] for r in queries)
    assert summary['optimizer'] == summary['Actor_Adam'] == summary['Value_Adam'] == summary['completed_paired_BC'] == 0
    assert summary['candidate'] is None and summary['proposal'] is None
    assert not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'counterexample_diagnosis.json', dict(state='COMPLETE_FROZEN_DIAGNOSIS_ONLY',
        source_rows=students+baselines, query_rows=queries, labels=statistics, coverage=coverage,
        labels_npz=dict(path=str(path), bytes=len(blob), sha256=sha(path)),
        original_autonomous8_are_historical_repeats=True, exact12_NOT_independent_new_data=True,
        teacher_labels_on_actual_student_history_from_h0=True, diagnosisrule=p['diagnosisrule'],
        optimizer=0, no_candidate_trained_or_transmitted=True, full_329=None,
        fresh16=None, reload78=None, six_gate=False, eight_started=False))
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0, optimizer=0, backward=0,
        physics=summary['physics'], commands=summary['commands'],
        all_full120_history_hidden_density_rewards_teacher_labels_and_integration_exact=True,
        all_autonomous8_original_policy_job_digests_and_integrations_exact=True,
        every12_exact_replay_all_arrays_and_integrations_exact=True,
        offline_teacher_labels_statistics_and_serialization_independently_exact_twice=True,
        prefix_checks=prefixes, all_safe=coverage['all_safe'], no_candidate_trained_or_transmitted=True,
        full_329=None, fresh16=None, reload78=None, six_gate=False, eight_started=False))
    verify(p, True)
    print('T469_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    assert command in {'preflight', 'main', 'audit'}, 'Only preflight/main/audit; registration and launch are separate'
    try:
        globals()[command]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{command}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
