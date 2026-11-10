"""Two fixed safeT255-reset recoveries from the rejected A325 histories.

Only the diagnostic teacher changes from the two failed T469 queries. Teacher
weights and recurrent execution change jointly; this is not a hidden-only
causal comparison. The A325 prefix remains frozen, and no optimizer is built.
Registration, launch and backup belong to the separate control program.
"""
from pathlib import Path
import importlib.util, json, os, shutil, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t471_reset_recovery'
DOC = ROOT/'stage/docs/overnight_20261110'
PRIOR = ROOT/'stage/runs/t469_counterexample_recovery'
spec = importlib.util.spec_from_file_location(
    'stage471_pinned468_runtime', ROOT/'stage/experiments/t468_offline_recovery_night/night_runtime.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
N = A.N
J, H, M, P, E, R, torch, np = A.J, A.H, A.M, A.P, A.E, A.R, A.torch, A.np
read, save, sha, digest, frozen = A.read, A.save, A.sha, A.digest, N.frozen
JOB_IDS = ['dev_task6_03', 'dev_task6_15']
TEACHER_KIND = 'safeT255reset'
PRIOR_TEACHER_KIND = 'T463_PPO327_actual_history_carry'
ROLE = 'MATCHED_SAFE_T255_RECOVERY2'
FIRST = 20
FULL_KEYS = {'states', 'actions', 'native_counts', 'next_states', 'poses',
             'next_poses', 'end_flags', 'rewards', 'history_poses',
             'history_goals', 'history_times', 'hidden_before',
             'initial_native_integration', 'final_native_integration'}


def prior_evidence(p):
    """Bind the reused full histories and failed controls to audited T469."""
    names = ['protocol.json', 'summary.json', 'audit.json', 'cases.jsonl',
             'replays.jsonl', 'SEALED.json']
    for name in names:
        path = str(PRIOR/name)
        assert path in p['inputs'] and sha(path) == p['inputs'][path], path
    old, summary, audit = (read(PRIOR/name) for name in names[:3])
    assert summary['state'] == 'COMPLETE' and audit['state'] == 'PASS'
    assert audit['all_full120_history_hidden_density_rewards_teacher_labels_and_integration_exact']
    assert audit['every12_exact_replay_all_arrays_and_integrations_exact']
    assert not summary['all_safe'] and not summary['six_gate']
    assert p['learning_initial'] == old['learning_initial']
    assert p['research_initial'] == old['research_initial']
    assert p['base_actor_digest'] == old['base_actor_digest']
    source = [r for r in summary['student_rows'] if r['job_id'] in JOB_IDS]
    failed = [r for r in summary['query_rows'] if r['job_id'] in JOB_IDS]
    assert p['source_rows'] == source and p['prior_failed_queries'] == failed
    assert [r['job_id'] for r in source] == [r['job_id'] for r in failed] == JOB_IDS
    assert all(not r['success'] and r['collision'] for r in source+failed)
    for row in source+failed:
        assert str(row['path']) in p['inputs']
        assert row['sha256'] == p['inputs'][str(row['path'])] == sha(row['path'])
        assert row['policy_hash'] == p['learning_initial']['policy_hash']
        assert row['loaded_model_sha256'] == p['learning_initial']['sha256']
    for row in failed:
        assert row['teacher'] and row['teacher_first'] == FIRST
        assert row['teacher_kind'] == PRIOR_TEACHER_KIND
        assert row['teacher_source_SHA256'] == old['teacher_model']['sha256']
    sealed = read(PRIOR/'SEALED.json')
    assert sealed['all_scientific_processes_stopped']
    assert all(not Path('/proc', str(pid)).exists() for pid in sealed['scientific_pids'])
    return old, source, failed


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert p['out'] == str(OUT)
    assert [j['id'] for j in p['diagnostic_jobs']] == JOB_IDS
    assert p['teacher_first'] == FIRST
    assert all(j.get('teacher_first', FIRST) == FIRST for j in p['diagnostic_jobs'])
    assert p['max_bytes'] == 8*2**20 and p['max_physics'] == 100000
    assert p['max_commands'] == 48 and p['preflight_commands'] == 24
    assert p['full_bytes'] == 448*1024 and p['compact_bytes'] == 4096
    assert p['main_seconds'] == 600 and p['learning_seconds'] == 1
    assert p['audit_seconds'] == 300 and p['all_seconds'] == 1800
    assert M.used(OUT) <= p['max_bytes']
    assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for path, expected in (p['inputs'] if full else p['protected_files']).items():
        assert sha(path) == expected, path
    for name in ['learning_initial', 'research_initial']:
        assert sha(p[name]['path']) == p[name]['sha256']
    if full:
        old, source, failed = prior_evidence(p)
        night = read(ROOT/'stage/runs/t468_offline_recovery_night/protocol.json')
        for key in ['absolute_learning_stop', 'absolute_native_stop', 'absolute_all_stop']:
            assert p[key] == old[key] == night[key]
        assert 2*sum(j['case']['step_cap'] for j in p['diagnostic_jobs']) <= p['max_physics']
        for job, original, control in zip(p['diagnostic_jobs'], source, failed):
            expected = dict(original['request']['job'], teacher_first=FIRST)
            assert dict(job, teacher_first=FIRST) == expected == control['request']['job']


def teacher(p, job):
    assert job['id'] in JOB_IDS and job.get('teacher_first', FIRST) == FIRST
    q = E.safe_initial(E.N.science(p), p).eval().requires_grad_(False)
    return q, TEACHER_KIND, digest(q.state_dict())


def worker(pipe, index, p, counts):
    # The pinned worker keeps native clock, history construction and student
    # carry active even after handover. Its safe branch uses float32 packets.
    A.teacher = teacher
    A.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for index in range(8):
                parent, child_pipe = ctx.Pipe()
                child = ctx.Process(target=worker, args=(child_pipe, index, p, counts))
                child.start(); child_pipe.close()
                self.children.append(child); self.pipes.append(parent)
            self.ready = [self.receive(index) for index in range(8)]
            assert all(row['state'] == 'READY' for row in self.ready)
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
    torch.set_grad_enabled(False)


def cpu_fixture(p):
    """Two independent frozen teachers label each real first21 history twice."""
    result = []
    for row in p['source_rows']:
        a = A.arrays(row)
        assert len(a['states']) > FIRST and a['states'].dtype == np.float32
        runs, hashes = [], []
        for _ in range(2):
            q, kind, hsh = teacher(p, row['request']['job'])
            assert kind == TEACHER_KIND and all(v.device.type == 'cpu' for v in q.parameters())
            assert all(not v.requires_grad for v in q.parameters())
            labels, hidden = [], None
            for x in a['states'][:FIRST+1]:
                action, hidden = A.teacher_command(q, kind, x, hidden)
                assert hidden is None and action.dtype == torch.float32
                labels.append(action[0].numpy().copy())
            labels = np.asarray(labels, np.float32)
            assert labels.shape == (FIRST+1, 2) and np.isfinite(labels).all() and np.abs(labels).max() <= 1
            runs.append(labels); hashes.append(hsh)
        assert hashes[0] == hashes[1] and np.array_equal(runs[0], runs[1])
        result.append(dict(job_id=row['job_id'], source_sha256=row['sha256'],
                           decisions=FIRST+1, teacher_state_sha256=hashes[0],
                           labels_digest=digest(runs[0]), float32_packet=True))
    return result


def preflight():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime(); readonly_cpu()
    q = frozen(p, p['learning_initial']['path']).eval().requires_grad_(False)
    assert q.phash() == p['learning_initial']['policy_hash']
    assert N.base_digest(q) == p['base_actor_digest']
    fixture = cpu_fixture(p)
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'; folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=min(time.time()+180, p['absolute_native_stop']), max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['learning_initial']['path'], job=job, seed=None,
                                     cancel_probe=True) for job in p['old8_jobs']])
        assert len(replies) == 8 and all(row['state'] == 'CANCEL_PASS' for row in replies)
        assert sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24, optimizer=0,
        Actor_Adam=0, Value_Adam=0, backward=0, frozen_A325_hash_and_A328_base_exact=True,
        actual_source_first21_teacher_fixture=fixture, safeT255reset_CPU_batch1_doublepass_exact=True,
        eight_workers_cancelled_before_first_native=True, all_workers_stopped=True,
        new_recovery_NOT_executed_yet=True))
    print('T471_PREFLIGHT_PASS', flush=True)


def prefix_check(query, source):
    assert query['teacher_first'] == FIRST and query['job_id'] == source['job_id']
    A.prefix_exact(query, source, FIRST)
    return dict(job_id=query['job_id'], teacher_first=FIRST, prefix_exact=True,
                source_path=source['path'], source_sha256=source['sha256'])


def main():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime(); readonly_cpu(); E.L.install_stop_handlers(); rt = M.Runtime(p)
    status, error, queries, coverage = 'FAILED', None, [], None
    try:
        rt.pool = Pool(dict(p, native_stop_epoch=min(rt.deadline, p['absolute_native_stop'])), rt.counts, rt.guard)
        jobs = [dict(job, teacher_first=FIRST) for job in p['diagnostic_jobs']]
        values = rt.request(jobs, p['learning_initial'], ROLE, full=True, query=True)
        queries = [row for a, row in values]
        rt.request(jobs, p['learning_initial'], ROLE+'_EXACT', full=True, references=queries, query=True)
        prefixes = [prefix_check(query, source) for query, source in zip(queries, p['source_rows'])]
        coverage = dict(all_safe=all(row['success'] and not row['collision'] for row in queries),
            teacher_results=R.M.tally(queries), prefixes=prefixes, teacher_new_queries=2,
            exact_replays=2, teacher_first=FIRST, source_full_rows_reused=2,
            teacher_weights_and_recurrent_execution_change_jointly=True,
            hidden_only_causal_comparison=False, prior_failed_queries=p['prior_failed_queries'])
        save(OUT/'coverage_gate.json', coverage)
        save(OUT/'support_rows.json', dict(source_rows=p['source_rows'], student_rows=p['source_rows'],
            prior_failed_queries=p['prior_failed_queries'], query_rows=queries, recovery_rows=queries))
        status = 'COMPLETE'  # A failed recovery is a completed fixed diagnostic.
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, raw_preserved=True, no_restart=True, optimizer=0))
    finally:
        if rt.pool: rt.pool.close(); rt.commands += 8
        save(OUT/'summary.json', dict(state=status, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            source_rows=p['source_rows'], student_rows=p['source_rows'], prior_failed_queries=p['prior_failed_queries'],
            query_rows=queries, recovery_rows=queries, coverage=coverage,
            all_safe=None if coverage is None else coverage['all_safe'], optimizer=0,
            Actor_Adam=0, Value_Adam=0, new_PPO=0, new_Value=0, backward=0,
            completed_paired_BC=0, candidate=None, proposal=None, teacher_executed_native_queries=len(queries),
            reused_sources_NOT_new_independent_samples=True, exact_replays_NOT_independent=True,
            hidden_only_causal_comparison=False, full_329=None, fresh16=None,
            reload78=None, six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=status, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN); rt.lock.close()
    assert status == 'COMPLETE', error
    print('T471_MAIN_COMPLETE', flush=True)


def audit_student(p, row):
    a = A.arrays(row)
    assert FULL_KEYS <= set(a) and a['states'].shape[1] == 120
    assert a['states'].dtype == np.float32
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    assert int(a['native_counts'].sum()) == row['physics'] and (a['native_counts'] > 0).all()
    assert not a['stochastic'].any() and not row['administrative_censor']
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
    assert [bool(v) for v in a['end_flags'][-1]] == [row['collision'], row['success'], row['timeout']]
    assert digest(a['final_native_integration']) == row['integration_sha256']
    assert np.array_equal(a['rewards'], a['environment_rewards'])
    assert row['loaded_model_sha256'] == p['learning_initial']['sha256']
    assert row['policy_hash'] == p['learning_initial']['policy_hash']
    assert row['request']['model'] == p['learning_initial']['path']
    assert row['seed'] is None and row['request']['seed'] is None
    for value in a.values():
        if np.issubdtype(value.dtype, np.number): assert np.isfinite(value).all()
    P.frozen = frozen
    try:
        M.audit_full_history(dict(p, teacher_first=FIRST), row, a)
    finally:
        P.frozen = H.ORIGINAL_FROZEN
    return a


def audit():
    p = read(OUT/'protocol.json'); verify(p, True); summary = read(OUT/'summary.json')
    assert summary['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    setup_runtime(); readonly_cpu()
    generations = read(OUT/'worker_generations.json')
    assert len(generations) == 1 and len(generations[0]['workers']) == 8
    ids = [summary['pid']]+[pid for group in generations for pid in group['workers']]
    ids += [pid for group in read(OUT/'preflight_native/worker_generations.json') for pid in group['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert len(rows) == len(replays) == 2
    assert [row['job_id'] for row in rows] == JOB_IDS
    assert all(row['role'] == ROLE for row in rows)
    assert summary['physics'] == sum(row['physics'] for row in rows+replays) <= p['max_physics']
    assert summary['commands'] == p['preflight_commands']+16+len(rows)+len(replays) == 44 <= p['max_commands']
    assert summary['query_rows'] == summary['recovery_rows'] == rows
    assert summary['source_rows'] == summary['student_rows'] == p['source_rows']
    assert summary['prior_failed_queries'] == p['prior_failed_queries']
    q = frozen(p, p['learning_initial']['path']).eval().requires_grad_(False)
    assert q.phash() == p['learning_initial']['policy_hash'] and N.base_digest(q) == p['base_actor_digest']
    old, sources, failed = prior_evidence(p)
    old_cases = list(map(json.loads, (PRIOR/'cases.jsonl').read_text().splitlines()))
    old_replays = list(map(json.loads, (PRIOR/'replays.jsonl').read_text().splitlines()))
    for source, control in zip(sources, failed):
        for row in [source, control]:
            assert row in old_cases
            matches = [r for r in old_replays if r['reference_path'] == row['path']]
            assert len(matches) == 1
            replay = matches[0]
            assert replay['exact'] and not replay['independent_new_sample']
            for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics']:
                assert replay[key] == row[key]
            audit_student(p, row)
        assert not source['teacher'] and not A.arrays(source)['ownership'].any()
        prefix_check(control, source)
        # Independently reproduce the original327 control on its actual history.
        carry = frozen(p, old['teacher_model']['path']).eval().requires_grad_(False)
        assert carry.phash() == old['teacher_model']['policy_hash']
        a, hidden, labels = A.arrays(control), None, []
        for x in a['states']:
            command, _, hidden, _ = carry.step(torch.tensor(x[None]), hidden)
            labels.append(command[0].numpy().copy())
        assert np.array_equal(np.asarray(labels, np.float32), a['teacher_actions'])
    teacher_hashes = []
    for row in rows:
        a = audit_student(p, row)
        assert row['teacher'] and row['teacher_first'] == FIRST
        assert row['teacher_kind'] == TEACHER_KIND and not row['teacher_hidden_burns_actual_history_from_h0']
        expert, kind, hsh = teacher(p, row['request']['job'])
        assert row['teacher_source_SHA256'] == hsh
        labels, hidden = [], None
        for x in a['states']:
            command, hidden = A.teacher_command(expert, kind, x, hidden)
            assert hidden is None and command.dtype == torch.float32
            labels.append(command[0].numpy().copy())
        assert a['teacher_actions'].dtype == np.float32
        assert np.array_equal(np.asarray(labels, np.float32), a['teacher_actions'])
        teacher_hashes.append(hsh)
    assert len(set(teacher_hashes)) == 1
    bypath = {row['path']: row for row in rows}
    for replay in replays:
        original = bypath[replay['reference_path']]
        assert replay['role'] == ROLE+'_EXACT' and replay['job_id'] == original['job_id']
        assert replay['request'] == original['request']
        for key in ['array_digest', 'array_hashes', 'integration_sha256', 'physics', 'policy_hash',
                    'loaded_model_sha256', 'teacher', 'teacher_first', 'teacher_kind',
                    'teacher_source_SHA256', 'success', 'collision', 'timeout']:
            assert replay[key] == original[key], key
        assert replay['exact'] and replay['independent_new_sample'] is False
        # The replay writes the same full path only after exact byte comparison.
        audit_student(p, dict(original, **replay))
    prefixes = [prefix_check(query, source) for query, source in zip(rows, sources)]
    coverage = read(OUT/'coverage_gate.json')
    assert coverage == summary['coverage'] and coverage['prefixes'] == prefixes
    assert coverage['prior_failed_queries'] == failed
    assert coverage['all_safe'] == summary['all_safe'] == all(row['success'] and not row['collision'] for row in rows)
    assert read(OUT/'support_rows.json') == dict(source_rows=sources, student_rows=sources,
        prior_failed_queries=failed, query_rows=rows, recovery_rows=rows)
    assert cpu_fixture(p) == read(OUT/'preflight.json')['actual_source_first21_teacher_fixture']
    for key in ['optimizer', 'Actor_Adam', 'Value_Adam', 'new_PPO', 'new_Value', 'backward', 'completed_paired_BC']:
        assert summary[key] == 0
    assert summary['candidate'] is None and summary['proposal'] is None
    assert summary['full_329'] is None and summary['fresh16'] is None and summary['reload78'] is None
    assert not summary['six_gate'] and not summary['eight_started'] and not summary['hidden_only_causal_comparison']
    assert not (OUT/'optimizer_calls.jsonl').exists()
    verify(p, True)
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0, optimizer=0, backward=0,
        physics=summary['physics'], commands=summary['commands'], all_safe=coverage['all_safe'],
        all_full120_history_hidden_density_rewards_teacher_labels_and_integration_exact=True,
        both_exact_replays_all_arrays_and_integrations_exact=True,
        T469_source_full_SHA_and_original_audit_and_failed327_carry_labels_exact=True,
        frozen_A325_and_original_A328_base_SHA_exact=True, safeT255reset_teacher_state_sha256=teacher_hashes[0],
        first20_student_actions_native_clock_history_hidden_and_initial_integration_exact=True,
        every_safeT255reset_label_CPU_float32_batch1_exact=True, prefix_checks=prefixes,
        teacher_weights_and_recurrent_execution_change_jointly=True, hidden_only_causal_comparison=False,
        no_candidate_trained_or_transmitted=True, full_329=None, fresh16=None, reload78=None,
        six_gate=False, eight_started=False))
    print('T471_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    assert command in {'preflight', 'main', 'audit'}, 'Only preflight/main/audit; registration and launch are separate'
    try:
        globals()[command]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{command}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
