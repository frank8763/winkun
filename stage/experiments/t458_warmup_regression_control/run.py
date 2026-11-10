"""Frozen warmup control separating pre-existing and later PPO regressions.

No training or model search. Original T455/T456/T457 results remain unchanged.
All comparisons use consumed development tasks, never the fresh holdout.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t458_warmup_regression_control'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage458_pinned449', ROOT/'stage/experiments/t449_offline_joint_recovery/run.py')
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
    first = dict(model=s['models'][0])
    assert Path(first['model']['path']).name == 'warmup.pt.gz'
    assert len(base['panel_jobs']) == 11
    latest = ROOT/'stage/runs/t457_balanced_task_ppo'
    assert read(latest/'audit.json')['state'] == 'PASS'
    assert read(DOC/'T457_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(latest/'SEALED.json')['scientific_pids'])
    assert not Path('/proc', str(read(latest/'driver_started.json')['pid'])).exists()
    keys = ['source_protocol', 'initial_model', 'student_start_model', 'student_start_sha256',
        'raw_sigma', 'phase', 'relay_stage', 'input_mean', 'input_scale', 'actor_seed', 'value_seed',
        'old275_jobs', 'main_jobs', 'panel_jobs', 'initial_success_ids', 'failed_job']
    p = {k: copy.deepcopy(base[k]) for k in keys}
    previous = list(map(json.loads, (prior/'cases.jsonl').read_text().splitlines()))
    p.update(id='T458_FROZEN_WARMUP_FULL329_CONTROL', out=str(OUT),
        research_model=first['model'], previous_panel=[r for r in previous if r['role'] == 'WARMUP_PANEL'],
        protected_files={**base['protected_files'], first['model']['path']: first['model']['sha256']},
        max_bytes=8*2**20, full_bytes=448*1024, compact_bytes=4096,
        max_physics=7000000, max_commands=400, preflight_commands=24,
        main_seconds=1800, learning_seconds=1800, audit_seconds=300, backup_seconds=600, all_seconds=3000,
        native_training=0, Actor_updates=0, Value_updates=0, optimizers=0,
        six_gate=False, fresh16=None, reload78=None, eight_started=False,
        mechanism='Freeze the exact common T455/T457 warmup state before PPO and evaluate original329 development tasks. Identify which failures already existed after128BC versus later PPO, without teacher or learning.',
        selection='Posthoc missing-stage control, not selection of a best checkpoint or independent training replicate. All329 consumed development, no fresh navigation holdout.',
        comparison_rows={
            'T455_PPO0_via_T456': str(ROOT/'stage/runs/t456_first_ppo_acceptance/cases.jsonl'),
            'T455_PPO1': str(prior/'cases.jsonl'),
            'T457_balanced_PPO': str(latest/'cases.jsonl')},
        original_T455_fixed_final_rejection_preserved=True, complete_fresh16_reload78_NOT_RESERVED=True,
        user_authorization='Nov10 user inspect then think and start next step: fill the missing frozen full329 warmup control before choosing a new learning mechanism; zero new training.')
    p['worst_physics'] = sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T458_PLAN.md'), str(DOC/'T458_PRIMARY_SOURCES.md'), str(DOC/'T457_FINDINGS.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'),
        str(prior/'cases.jsonl'), str(prior/'SEALED.json'), first['model']['path'],
        str(latest/'protocol.json'), str(latest/'summary.json'), str(latest/'audit.json'),
        str(latest/'SEALED.json'), str(DOC/'T457_FINAL_REPORT.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T457.json')}
    files |= set(p['comparison_rows'].values())
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    storage = 329*p['compact_bytes']+8*p['full_bytes']+3*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T458_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        storage_worst=storage, global_full=M.used()-M.used(OUT)+p['max_bytes'],
        max_physics=p['max_physics'], worst_physics=p['worst_physics'],
        primary_native_episodes=329, exact_replays=0, historical_panel_comparisons=11,
        maximum_all_commands=369, max_commands=400, new_training_updates=0,
        main_seconds=1800, audit_seconds=300, all_seconds=3000,
        estimated_main_minutes=[7, 15], hard_cap_NOT_queued_duration=True,
        fresh16_reload78_NOT_reserved=True, original_models_copied_or_deleted=0))
    print('T458_REGISTERED', p['worst_physics'], flush=True)


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
    assert len(p['previous_panel']) == 11
    assert sum(r['success'] for r in p['previous_panel']) == 10
    assert all(r['loaded_model_sha256'] == p['research_model']['sha256'] for r in p['previous_panel'])
    fixture = [r for r in map(json.loads, Path(p['comparison_rows']['T457_balanced_PPO']).read_text().splitlines())
               if r['role'] == 'FINAL329']
    own = stage_comparison(p, fixture)['later']['T457_balanced_PPO']
    assert not own['lost_after_warmup'] and not own['repaired_after_warmup']
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
        eight_workers_cancelled_and_stopped=True, no_teacher_or_sampling=True,
        paired_stage_join_self_difference_zero=True, historical_10_success_1_failure_locked=True))
    print('T458_PREFLIGHT_PASS', flush=True)


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
        candidate, raw = M.evaluate(rt, p, p['research_model'], 'FROZEN_WARMUP_329')
        previous = {r['job_id']: r for r in p['previous_panel']}
        repeated = []
        for r in raw:
            if r['job_id'] in previous:
                old = previous[r['job_id']]
                for key in ['array_digest', 'array_hashes', 'integration_sha256']:
                    assert r[key] == old[key], (r['job_id'], key)
                for key in ['success', 'collision', 'timeout', 'physics']:
                    assert r[key] == old[key], (r['job_id'], key)
                repeated.append(r['job_id'])
        assert len(repeated) == 11
        candidate['preserves_initial328'] = set(p['initial_success_ids']) <= set(candidate['success_ids'])
        candidate['repairs00'] = p['failed_job']['id'] in candidate['success_ids']
        candidate['development_admitted'] = len(candidate['success_ids']) == 329
        save(OUT/'historical_panel_comparison.json', dict(exact_ids=repeated,
            physical_repeats_not_new_independent_examples=11, new_nonpanel_policy_job_evaluations=318))
        save(OUT/'stage_comparison.json', stage_comparison(p, raw))
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
            candidate=candidate, new_training_updates=0, posthoc_missing_stage_control=True,
            original_T455_final_rejection_unchanged=True, fresh16=None, reload78=None,
            six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=state, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error
    print('T458_MAIN_COMPLETE', flush=True)


def stage_comparison(p, rows):
    current = {r['job_id']: r for r in rows}
    warm_success = {k for k, r in current.items() if r['success']}
    warm_failure = set(current)-warm_success
    original = set(p['initial_success_ids'])
    result = dict(warmup_success_ids=sorted(warm_success), warmup_failure_ids=sorted(warm_failure),
        original328_lost_already_at_warmup=sorted(original-warm_success),
        original00_repaired_at_warmup=p['failed_job']['id'] in warm_success,
        all_comparisons_consumed_development=True,
        deterministic_stage_change_NOT_multi_seed_causal_generalization=True, later={})
    for name, path in p['comparison_rows'].items():
        prior_rows = [r for r in map(json.loads, Path(path).read_text().splitlines())
                      if r['job_id'] in current and r['role'] in ['FINAL329', 'FROZEN_FIRST_PPO_329']]
        assert len(prior_rows) == len({r['job_id'] for r in prior_rows}) == 329
        success = {r['job_id'] for r in prior_rows if r['success']}
        result['later'][name] = dict(success=len(success),
            lost_after_warmup=sorted(warm_success-success),
            repaired_after_warmup=sorted(success-warm_success),
            failures_shared_with_warmup=sorted(warm_failure-success))
    return result


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
    assert stage_comparison(p, rows) == read(OUT/'stage_comparison.json')
    assert sha(p['research_model']['path']) == p['research_model']['sha256']
    save(OUT/'audit.json', dict(state='PASS_FROZEN_COMPACT_EVIDENCE', pid=os.getpid(),
        new_native=0, optimizers=0, same_frozen_model_every_case_verified=True,
        compact_actions_termination_and_native_count_SHA_exact=True,
        complete120_hidden_and_native_scene_history_NOT_reconstructed=True,
        final_valid=329, wide54=wide, old275=old, physics=s['physics'], commands=s['commands'],
        historical_panel11_success_and_failure_exact=True,
        stage_by_stage_success_ID_differences_recomputed=True,
        fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T458_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T458',
            '--why=Frozen warmup Stage full development control', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, native_stop=now+p['main_seconds'],
        all_stop=now+p['all_seconds'], protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T458_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
