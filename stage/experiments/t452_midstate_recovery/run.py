"""Frozen student-prefix recovery audit; no optimizer, grid, or deployment teacher."""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t452_midstate_recovery'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage452_pinned449', ROOT/'stage/experiments/t449_offline_joint_recovery/run.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
P, R, E, torch, np = M.P, M.R, M.E, M.torch, M.np
read, save, sha, digest = P.read, P.save, P.sha, P.digest


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes']
    assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def setup_runtime():
    M.OUT = OUT
    M.verify = verify
    M.Pool = Pool
    M.setup_runtime()


def prefix_for_worker(p, index):
    # A fixed worker binding, not a deployment router. All requests in this
    # frozen diagnostic use the same policy/job; only teacher takeover differs.
    return p['teacher_steps'][index % 3]


def worker(pipe, index, p, counts):
    M.worker(pipe, index, dict(p, teacher_first=prefix_for_worker(p, index)), counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                child = ctx.Process(target=worker, args=(b, i, p, counts))
                child.start()
                b.close()
                self.children.append(child)
                self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(v['state'] == 'READY' for v in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            rows = read(f) if f.exists() else []
            rows.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, rows)
        except BaseException:
            self.close()
            raise


def check_prefix(base, a, first):
    assert len(base['states']) > first and len(a['states']) > first
    keys = []
    for key, value in base.items():
        if np.ndim(value) and len(value) == len(base['states']):
            assert np.array_equal(a[key][:first], value[:first]), key
            keys.append(key)
    for key in ['states', 'poses', 'hidden_before', 'history_poses', 'history_goals']:
        assert np.array_equal(a[key][first], base[key][first]), key
    for key in ['task_goal_xy', 'scene_sha256']:
        assert np.array_equal(a[key], base[key]), key
    assert int(a['native_counts'][:first].sum()) == first*100
    assert not a['ownership'][:first].any() and a['ownership'][first:].all()
    return dict(first=first, simulation_seconds=first*.2, equal_prefix_keys=keys,
                same_takeover_observation_pose_hidden_and_history=True,
                original_clock_and_goal_preserved=True)


def prepare():
    assert not OUT.exists(), 'Never restart a registered/ended directory'
    prior = ROOT/'stage/runs/t451_single_failure_recovery'
    base = read(prior/'protocol.json')
    assert read(prior/'audit.json')['state'] == 'PASS'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert read(DOC/'T451_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    keys = ['source_protocol', 'initial_model', 'student_start_model', 'student_start_sha256',
            'raw_sigma', 'phase', 'relay_stage', 'input_mean', 'input_scale', 'actor_seed', 'value_seed']
    p = {k: copy.deepcopy(base[k]) for k in keys}
    p.update(id='T452_FROZEN_MID_STUDENT_STATE_RECOVERY', out=str(OUT),
             research_initial=base['research_initial'], failed_job=base['failed_job'],
             teacher_steps=[20, 60, 120], teacher_first=20,
             max_bytes=12*2**20, full_bytes=448*1024, compact_bytes=4096,
             max_physics=200000, max_commands=64, preflight_commands=24,
             main_seconds=900, learning_seconds=900, audit_seconds=600,
             backup_seconds=600, all_seconds=2400,
             Actor_Adam=0, V_Adam=0, PPO_updates=0, BC_updates=0,
             maximum_primary_trajectories=4, maximum_exact_replays=4,
             six_gate=False, fresh16=None, reload78=None, eight_started=False,
             parent_cases=str(prior/'cases.jsonl'),
             mechanism='Frozen328/329 student actual4/12/24s prefixes, T255 base_actions recovery to unchanged old00 final goal; no parameter updates.',
             scope='Sole remaining OLD5m regression dev_task6_00, not a new six-metre acceptance. Mixed teacher success is not autonomous student success.',
             selection='All three fixed prefixes executed regardless of outcome; no replacement times/seeds/teachers. Report verified points only, not a recovery interval.',
             deployment='Unchanged single causal120 GRU64 Actor drive/steer; teacher only inside this training-data diagnostic.',
             user_authorization='Nov10 user start next step after end-to-end plan: independent bounded frozen midstate recovery diagnosis, not T451 restart.')
    old = [json.loads(x) for x in (prior/'cases.jsonl').read_text().splitlines()]
    p['parent_reference'] = next(r for r in old if r['role'] == 'INITIAL_FAILED00')
    assert not p['parent_reference']['success'] and p['parent_reference']['full']
    p['protected_files'] = copy.deepcopy(base['protected_files'])
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T452_PLAN.md'), str(DOC/'END_TO_END_CONTINUATION_SOURCES.md'),
              str(prior/'audit.json'), str(prior/'summary.json'), str(prior/'cases.jsonl'),
              str(prior/'protocol.json'), p['parent_reference']['path'], p['research_initial']['path']}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['worst_physics'] = 8*p['failed_job']['case']['step_cap']
    assert p['worst_physics'] <= p['max_physics']
    storage = 4*p['full_bytes'] + 8*p['full_bytes'] + 5*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] <= 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T452_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED',
         own_max_bytes=p['max_bytes'], storage_worst=storage,
         global_before=M.used()-M.used(OUT), global_full=M.used()-M.used(OUT)+p['max_bytes'],
         primary=4, exact=4, maximum_all_commands=48, max_commands=64,
         worst_physics=p['worst_physics'], max_physics=200000,
         Actor_Adam=0, V_Adam=0, main_seconds=900, audit_seconds=600, all_seconds=2400,
         estimated_main_minutes=[2, 5], hard_cap_is_NOT_queued_duration=True,
         originals_deleted=0, copied_model_bytes=0))
    print('T452_REGISTERED', p['worst_physics'], flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    R.C.ban_learning()
    q = P.frozen(p, p['research_initial']['path'])
    before = digest(P.checkpoint(p['research_initial']['path'])['joint_state'])
    assert before == p['research_initial']['joint_hash']
    reference = p['parent_reference']
    a = E.N.S.arrays(reference['path'])
    M.audit_full_history(p, reference, a)
    assert len(a['states']) > max(p['teacher_steps'])
    expert = E.safe_initial(E.N.science(p), p).eval().requires_grad_(False)
    with torch.no_grad():
        commands = expert.base_actions(torch.tensor(a['states'][[20, 60, 120]])).float().numpy()
    assert np.isfinite(commands).all() and np.abs(commands).max() <= 1
    assert [prefix_for_worker(p, i) for i in range(3)] == p['teacher_steps']
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'
    folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['research_initial']['path'], job=p['failed_job'], seed=None, cancel_probe=True) for _ in range(8)])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally:
        pool.close()
    assert before == digest(P.checkpoint(p['research_initial']['path'])['joint_state'])
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24,
         Actor_Adam=0, V_Adam=0, historical_parent_full_history_exact=True,
         teacher_commands_on_actual_student_states=commands.tolist(),
         prefix_steps=p['teacher_steps'], worker_binding=[prefix_for_worker(p, i) for i in range(8)],
         eight_workers_cancelled_and_stopped=True, model_joint_SHA_unchanged=True))
    print('T452_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_ADAM', flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime()
    E.N.offline()
    R.C.ban_learning()
    E.L.install_stop_handlers()
    rt = M.Runtime(p)
    state, error, outcomes, baseline = 'FAILED', None, [], None
    try:
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        original = rt.request([p['failed_job']], p['research_initial'], 'FROZEN_STUDENT_CONTROL', full=True)
        baseline = original[0][1]
        parent = p['parent_reference']
        for key in ['array_digest', 'array_hashes', 'integration_sha256']:
            assert baseline[key] == parent[key], key
        assert not baseline['success']
        rt.request([p['failed_job']], p['research_initial'], 'FROZEN_STUDENT_CONTROL_EXACT', full=True, references=[baseline])
        values = rt.request([p['failed_job']]*3, p['research_initial'], 'MIDSTATE_RECOVERY', full=True, query=True)
        for (a, r), first in zip(values, p['teacher_steps']):
            assert r['teacher_first'] == first
            prefix = check_prefix(original[0][0], a, first)
            outcomes.append(dict(first=first, seconds=first*.2, recovery_safe=bool(r['success'] and not r['collision']),
                                 row=r, prefix=prefix, student_success=False,
                                 student_owned_decisions=first, teacher_owned_decisions=len(a['states'])-first))
        save(OUT/'prefix_checks.json', outcomes)
        rt.request([p['failed_job']]*3, p['research_initial'], 'MIDSTATE_RECOVERY_EXACT', full=True, references=[r for a, r in values], query=True)
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, physics=sum(rt.counts), commands=rt.commands))
    finally:
        if rt.pool:
            rt.pool.close()
            rt.commands += 8
        summary = dict(state=state, pid=os.getpid(), elapsed_s=time.time()-rt.start,
                       physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'],
                       baseline=baseline, outcomes=outcomes, error=error,
                       Actor_Adam=0, V_Adam=0, PPO_updates=0, BC_updates=0,
                       six_gate=False, old275=None, wide54=None, fresh16=None, reload78=None,
                       eight_started=False, all_native_workers_stopped=True)
        save(OUT/'summary.json', summary)
        save(OUT/'status.json', dict(state=state, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error
    print('T452_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p, True)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    R.C.ban_learning()
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert len(rows) == 4 and len(replays) == 4
    assert s['physics'] == sum(r['physics'] for r in rows+replays)
    assert s['commands'] == 24+16+len(rows)+len(replays) == 48
    expert = E.safe_initial(E.N.science(p), p).eval().requires_grad_(False)
    base = E.N.S.arrays(rows[0]['path'])
    checks = []
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert digest(a) == r['array_digest']
        assert {k: digest(v) for k, v in a.items()} == r['array_hashes']
        assert int(a['native_counts'].sum()) == r['physics']
        M.audit_full_history(dict(p, teacher_first=r['teacher_first'] or p['teacher_first']), r, a)
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        if r['teacher']:
            with torch.no_grad():
                labels = expert.base_actions(torch.tensor(a['states'])).float().numpy()
            assert np.array_equal(labels, a['teacher_actions'])
            checks.append(check_prefix(base, a, r['teacher_first']))
    bypath = {r['path']: r for r in rows}
    for r in replays:
        old = bypath[r['reference_path']]
        for key in ['array_digest', 'array_hashes', 'integration_sha256']:
            assert r[key] == old[key], key
    safe = [r['teacher_first']*.2 for r in rows if r['teacher'] and r['success'] and not r['collision']]
    assert sha(p['research_initial']['path']) == p['research_initial']['sha256']
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0,
         Actor_Adam=0, V_Adam=0, all_full120_history_hidden_teacher_reward_and_SHA_exact=True,
         prefix_checks=checks, verified_safe_takeover_seconds=safe,
         verified_points_only_NOT_interval=True, autonomous_student_navigation_gain=False,
         historical_control_repeat_not_new_independent_sample=True,
         actual_primary=4, exact_replays=4, physics=s['physics'], commands=s['commands']))
    print('T452_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        child = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T452',
            '--why=Bounded frozen student midstate recovery diagnosis', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=child.pid, epoch=now, native_stop=now+p['main_seconds'],
         all_stop=now+p['all_seconds'], protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T452_ACTUALLY_LAUNCHED', child.pid, flush=True)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
