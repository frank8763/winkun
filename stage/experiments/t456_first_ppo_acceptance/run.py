"""Frozen full-development evaluation of the post-hoc first-PPO candidate.

No training or parameter search. The original T455 fixed-last result remains
rejected; fresh navigation holdout and independent78 are not consumed here.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t456_first_ppo_acceptance'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage456_pinned449', ROOT/'stage/experiments/t449_offline_joint_recovery/run.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
P, R, E, torch, np = M.P, M.R, M.E, M.torch, M.np
read, save, sha, digest = P.read, P.save, P.sha, P.digest


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def worker(pipe, index, p, counts):
    M.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start()
                b.close()
                self.children.append(c)
                self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(v['state'] == 'READY' for v in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            generations = read(f) if f.exists() else []
            generations.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, generations)
        except BaseException:
            self.close()
            raise


def prepare():
    assert not OUT.exists(), 'Never restart an existing directory'
    prior = ROOT/'stage/runs/t455_frozen_representation_ppo'
    base = read(prior/'protocol.json')
    s = read(prior/'summary.json')
    assert s['state'] == 'COMPLETE' and read(prior/'audit.json')['state'] == 'PASS'
    assert read(DOC/'T455_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc', str(read(prior/'driver_started.json')['pid'])).exists()
    first = s['rounds'][0]
    assert len(first['panel_success_ids']) == len(base['panel_jobs']) == 11
    keys = ['source_protocol', 'initial_model', 'student_start_model', 'student_start_sha256',
        'raw_sigma', 'phase', 'relay_stage', 'input_mean', 'input_scale', 'actor_seed', 'value_seed',
        'old275_jobs', 'main_jobs', 'panel_jobs', 'initial_success_ids', 'failed_job']
    p = {k: copy.deepcopy(base[k]) for k in keys}
    previous = list(map(json.loads, (prior/'cases.jsonl').read_text().splitlines()))
    p.update(id='T456_FROZEN_POSTHOC_FIRST_PPO_FULL329', out=str(OUT),
        research_model=first['model'], previous_panel=[r for r in previous if r['role'] == 'PPO_PANEL_0'],
        protected_files={**base['protected_files'], first['model']['path']: first['model']['sha256']},
        max_bytes=8*2**20, full_bytes=448*1024, compact_bytes=4096,
        max_physics=7000000, max_commands=400, preflight_commands=24,
        main_seconds=1800, learning_seconds=1800, audit_seconds=300, backup_seconds=600, all_seconds=3000,
        native_training=0, Actor_updates=0, Value_updates=0, optimizers=0,
        six_gate=False, fresh16=None, reload78=None, eight_started=False,
        mechanism='Freeze T455 firstPPO model after observed11/11 development panel; independently evaluate same329 original development tasks, no teacher, no learning.',
        selection='Posthoc research checkpoint choice from already consumed development11; not preregistered T455 final, not independent training replicate. No navigation holdout used for selection.',
        original_T455_fixed_final_rejection_preserved=True, complete_fresh16_reload78_NOT_RESERVED=True,
        user_authorization='Nov10 user inspect and continue next step: evaluate existing firstPPO all-center candidate before changing algorithm or adding training.')
    p['worst_physics'] = sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T456_PLAN.md'), str(DOC/'T456_PRIMARY_SOURCES.md'), str(DOC/'T455_FINDINGS.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'),
        str(prior/'cases.jsonl'), str(prior/'SEALED.json'), first['model']['path']}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    storage = 329*p['compact_bytes']+8*p['full_bytes']+3*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T456_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        storage_worst=storage, global_full=M.used()-M.used(OUT)+p['max_bytes'],
        max_physics=p['max_physics'], worst_physics=p['worst_physics'],
        primary_native_episodes=329, exact_replays=0, historical_panel_comparisons=11,
        maximum_all_commands=369, max_commands=400, new_training_updates=0,
        main_seconds=1800, audit_seconds=300, all_seconds=3000,
        estimated_main_minutes=[7, 15], hard_cap_NOT_queued_duration=True,
        fresh16_reload78_NOT_reserved=True, original_models_copied_or_deleted=0))
    print('T456_REGISTERED', p['worst_physics'], flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    R.C.ban_learning()
    q = P.frozen(p, p['research_model']['path'])
    assert q.phash() == p['research_model']['policy_hash']
    assert digest(P.checkpoint(p['research_model']['path'])['joint_state']) == p['research_model']['joint_hash']
    assert len({j['id'] for j in p['main_jobs']+p['old275_jobs']}) == 329
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'
    folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['research_model']['path'], job=j, seed=None, cancel_probe=True) for j in p['panel_jobs'][:8]])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, optimizers=0, commands=24,
        original_frozen_model_and_both_Adam_SHA_exact=True, unique_original_development_jobs=329,
        eight_workers_cancelled_and_stopped=True, no_teacher_or_sampling=True))
    print('T456_PREFLIGHT_PASS', flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime()
    E.N.offline()
    R.C.ban_learning()
    E.L.install_stop_handlers()
    rt = M.Runtime(p)
    state, error, candidate = 'FAILED', None, None
    try:
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        candidate, raw = M.evaluate(rt, p, p['research_model'], 'FROZEN_FIRST_PPO_329')
        previous = {r['job_id']: r for r in p['previous_panel']}
        repeated = []
        for r in raw:
            if r['job_id'] in previous:
                old = previous[r['job_id']]
                for key in ['array_digest', 'array_hashes', 'integration_sha256']:
                    assert r[key] == old[key], (r['job_id'], key)
                assert r['success'] and not r['collision']
                repeated.append(r['job_id'])
        assert len(repeated) == 11
        candidate['preserves_initial328'] = set(p['initial_success_ids']) <= set(candidate['success_ids'])
        candidate['repairs00'] = p['failed_job']['id'] in candidate['success_ids']
        candidate['development_admitted'] = len(candidate['success_ids']) == 329
        save(OUT/'historical_panel_comparison.json', dict(exact_ids=repeated,
            physical_repeats_not_new_independent_examples=11, new_nonpanel_policy_job_evaluations=318))
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, physics=sum(rt.counts), commands=rt.commands))
    finally:
        if rt.pool:
            rt.pool.close()
            rt.commands += 8
        save(OUT/'summary.json', dict(state=state, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            candidate=candidate, new_training_updates=0, posthoc_development_checkpoint_selection=True,
            original_T455_final_rejection_unchanged=True, fresh16=None, reload78=None,
            six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=state, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error
    print('T456_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p, True)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    assert len(rows) == len({r['job_id'] for r in rows}) == 329
    assert s['physics'] == sum(r['physics'] for r in rows)
    assert s['commands'] == 24+16+len(rows) == 369
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert {k: digest(v) for k, v in a.items()} == {k: r['array_hashes'][k] for k in a}
        assert r['loaded_model_sha256'] == p['research_model']['sha256']
        assert r['policy_hash'] == p['research_model']['policy_hash'] and r['seed'] is None and not r['teacher']
        assert int(a['native_counts'].sum()) == r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert [bool(v) for v in a['end_flags'][-1]] == [bool(r['collision']), bool(r['success']), bool(r['timeout'])]
        assert np.abs(a['actions']).max() <= 1 and not r['administrative_censor']
    wide, old = M.R.M.tally(rows[:54]), M.R.M.tally(rows[54:])
    assert wide == s['candidate']['wide54'] and old == s['candidate']['old275']
    assert sha(p['research_model']['path']) == p['research_model']['sha256']
    save(OUT/'audit.json', dict(state='PASS_FROZEN_COMPACT_EVIDENCE', pid=os.getpid(),
        new_native=0, optimizers=0, same_frozen_model_every_case_verified=True,
        compact_actions_termination_and_native_count_SHA_exact=True,
        complete120_hidden_and_native_scene_history_NOT_reconstructed=True,
        final_valid=329, wide54=wide, old275=old, physics=s['physics'], commands=s['commands'],
        historical_panel11_exact=True, fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T456_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T456',
            '--why=Frozen firstPPO complete Stage development acceptance', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, native_stop=now+p['main_seconds'],
        all_stop=now+p['all_seconds'], protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T456_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
