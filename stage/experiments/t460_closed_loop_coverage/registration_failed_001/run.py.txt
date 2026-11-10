"""One bounded current-student recovery and original-own-history coverage pilot.

Same T459 neural architecture and128BC. Only new validated recovery/retention
coverage changes; no new PPO, runtime teacher, planner, routing or waypoint.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t460_closed_loop_coverage'
DOC = ROOT/'stage/docs/review_20261110'
sys.path.insert(0, str(HERE))
import t460_cache as C
spec = importlib.util.spec_from_file_location('stage460_pinned459', ROOT/'stage/experiments/t459_nonlinear_residual_repair/run.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
M, P, R, E, D, F, torch, np = H.M, H.P, H.R, H.E, H.D, H.F, H.torch, H.np
read, save, sha, digest = H.read, H.save, H.sha, H.digest
make, state, load, export, frozen, step = H.make, H.state, H.load, H.export, H.frozen, H.step


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used() <= 2**30
    if (OUT/'cache_summary.json').exists():
        assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def worker(pipe, index, p, counts):
    P.frozen = frozen
    M.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start(); b.close()
                self.children.append(c); self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(r['state'] == 'READY' for r in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            rows = read(f) if f.exists() else []
            rows.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, rows)
        except BaseException:
            self.close(); raise


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def data(p, own_rows, recovery_rows):
    pp = copy.deepcopy(p)
    pp['own_rows'] = p['own_rows']+own_rows
    parent = H.ORIGINAL_FROZEN(p, p['research_initial']['path']).cuda()
    groups, own = D.anchors(pp, parent)
    corrections = [D.correction(r) for r in recovery_rows]
    assert len(groups) == 218 and len(own) == 15 and len(corrections) == 6
    return groups, own, corrections


def prepare():
    assert not OUT.exists(), 'Never restart'
    prior = ROOT/'stage/runs/t459_nonlinear_residual_repair'
    assert read(prior/'audit.json')['state'] == 'PASS'
    assert read(DOC/'T459_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc', str(read(prior/'driver_started.json')['pid'])).exists()
    base = read(prior/'protocol.json')
    s = read(prior/'summary.json')
    p = copy.deepcopy(base)
    oldrows = list(map(json.loads, (prior/'cases.jsonl').read_text().splitlines()))
    failures = sorted(r['job_id'] for r in oldrows if not r['success'])
    assert failures == ['dev_task6_00', 'dev_task6_03', 'dev_task6_09', 'dev_task6_14', 'dev_task6_21', 'dev_task6_24']
    jobs = {j['id']: j for j in p['old275_jobs']}
    p.update(id='T460_ACTUAL_CLOSED_LOOP_RETENTION_AND_RECOVERY_COVERAGE', out=str(OUT),
        query_model=s['proposal'], recovery_jobs=[jobs[k] for k in failures],
        preservation_jobs=[jobs[k] for k in failures if k != 'dev_task6_00'],
        query_parent_rows=[r for r in oldrows if r['job_id'] in failures],
        historical_parent_rows=[r for r in map(json.loads, Path(p['parent_cases']).read_text().splitlines())
                                if r['role'] == 'FINAL_recovery' and r['job_id'] in failures and r['success']],
        mechanism='Same original328 zero-residual1074 network/freshAdam128BC asT459; add5original successful actual perturbed histories to10own retention, and6currentT459failed-case4sec success-verified teacher recoveries; complete329 fixed candidate.',
        new_mechanism='Correct actual visited-state coverage gap in both preservation and recovery; keep network, optimizer, update count and loss formulas fixed.',
        training='128supervised residualAdam+128shadow,0newPPO/Value. Only after5parent histories match original native hash and6teacher recoveries allsafe+exact; recovery labels auxiliary imitation, not onpolicy.',
        comparison='Joint recovery and retention data-coverage pilot; fresh physical episodes, unequal data/compute versusT459, not isolated causal ablation, originality or safety certificate.',
        user_authorization='Nov10 user asks whyold5degrades and continue solve; one bounded actual-history coverage mechanism plus previously authorized GitHub-only unused cache eviction after fresh fullbinary verification.',
        max_bytes=16*2**20, max_physics=8000000, max_commands=432,
        zero_model_path=str(prior/'initial_zero.pt.gz'),
        teacher_first=20, maximum_teacher_trajectories=6, maximum_teacher_exact_replays=6,
        maximum_parent_preservation_trajectories=5, maximum_parent_exact_replays=5,
        maximum_new_training_episodes=0, maximum_exact_training_replays=0,
        preflight_discarded_Actor_Adam=2,
        original_parent_and_rejected_runs_unchanged=True,
        all_six_teacher_recoveries_must_be_safe_before_BC=True,
        final_selection='Fixed128th proposal complete329; preserveoriginal328 andrepair00 required. If teacher bundle fails, stop before BC; no substitute subset or more queries.',
        cache_remote_originals_preserved_not_all_local_raw_claim=True)
    assert p['research_initial'] == base['research_initial'] and p['residual_seed'] == base['residual_seed']
    for key in ['warmup_steps', 'repair_lr', 'raw_sigma', 'actor_trainable_parameters', 'base_actor_digest']:
        assert p[key] == base[key]
    assert len(p['historical_parent_rows']) == 5
    p['protected_files'] = {**base['protected_files'], s['proposal']['path']: s['proposal']['sha256'],
        p['zero_model_path']: sha(p['zero_model_path'])}
    files = set(base['inputs']) | set(p['protected_files']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T460_PLAN.md'), str(DOC/'T460_PRIMARY_SOURCES.md'), str(DOC/'T459_FINDINGS_AND_NEXT.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'), str(prior/'cases.jsonl'),
        str(DOC/'T459_FINAL_REPORT.json'), str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T459.json')}
    files |= {r['path'] for r in p['query_parent_rows']}
    p['remote_cache'] = C.candidates(files, 16*2**20)
    for key in ['receipt', 'manifest']:
        files.add(str(ROOT/p['remote_cache'][key]))
    assert not {str(ROOT/r['path']) for r in p['remote_cache']['rows']} & files
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    p['worst_physics'] = sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])+2*sum(j['case']['step_cap'] for j in p['recovery_jobs']+p['preservation_jobs'])
    assert p['worst_physics'] < p['max_physics']
    storage = 19*p['full_bytes']+329*p['compact_bytes']+p['model_bytes']+4*2**20
    assert storage <= p['max_bytes']
    assert M.used()-p['remote_cache']['conservative_released_bytes']+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p); save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T460_BUDGET.json', dict(state='REGISTERED_BEFORE_CACHE_AND_PREFLIGHT_NOT_LAUNCHED',
        own_max=p['max_bytes'], storage_worst=storage,
        global_full_after_cache=M.used()-M.used(OUT)-p['remote_cache']['conservative_released_bytes']+p['max_bytes'],
        cache_local_evicted_bytes_min=p['remote_cache']['conservative_released_bytes'], cache_files=len(p['remote_cache']['rows']),
        max_physics=p['max_physics'], worst_physics=p['worst_physics'], maximum_all_commands=391,
        main_seconds=p['main_seconds'], learning_seconds=p['learning_seconds'], audit_seconds=p['audit_seconds'], all_seconds=p['all_seconds'],
        estimated_main_minutes=[12, 30], hard_cap_NOT_queued_duration=True,
        BC_live_max=128, BC_shadow_max=128, fresh16_reload78_NOT_reserved=True,
        remote_cache_readback600seconds_before_launch_NOT_new_navigation=True))
    print('T460_REGISTERED', p['worst_physics'], len(p['remote_cache']['rows']), flush=True)


def cache_evict():
    p = read(OUT/'protocol.json'); verify(p, True); C.evict(p)


def cache_audit():
    p = read(OUT/'protocol.json'); C.audit(p); verify(p, True)


def preflight():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'cache_audit.json')['state'] == 'PASS_REFERENCE_AND_FRESH_REMOTE_ARCHIVE_RECEIPT'
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    a, b = make(p), make(p)
    initial = state(*a)
    zero = read(ROOT/'stage/runs/t459_nonlinear_residual_repair/preflight.json')['zero_model']
    assert a[0].phash() == zero['policy_hash'] and digest(initial) == zero['joint_hash']
    # Shape fixture only: six aliases and five repeated old anchors do not
    # masquerade as the uncollected actual teacher/preservation trajectories.
    fixture = data(p, p['own_rows'][:5], [p['initial_correction']]*6)
    va, vb = step(a, fixture), step(b, fixture)
    assert va == vb and digest(state(*a)) == digest(state(*b))
    load(a, initial); load(b, initial)
    assert digest(state(*a)) == digest(initial) == digest(state(*b))
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'; folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=zero['path'], job=j, seed=None, cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally: pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24, discarded_synthetic_Adam=2,
        zero_model=zero, same_T459_network_seed_optimizer_and_initial_full_state=True,
        six_recovery_fifteen_own_loss_fixture_NOT_collected_data=True,
        initial_retention=va, live_shadow_and_rollback_exact=True,
        eight_workers_cancelled_and_stopped=True, actual_teacher_gate_NOT_yet_executed=True))
    print('T460_PREFLIGHT_PASS', flush=True)


def main():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime(); E.N.offline(); E.L.install_stop_handlers(); rt = M.Runtime(p)
    status, error, candidate, proposal = 'FAILED', None, None, None
    gate, own, queries, completed = None, [], [], 0
    obj = shadow = initial = None
    try:
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        values = rt.request(p['preservation_jobs'], p['research_initial'], 'PARENT_OWN_PERTURBED5', full=True)
        H.F.check_parent_reproduction(values, {r['job_id']: r for r in p['historical_parent_rows']})
        own = [r for a, r in values]
        assert all(r['success'] and not r['collision'] for r in own)
        rt.request(p['preservation_jobs'], p['research_initial'], 'PARENT_OWN_PERTURBED5_EXACT', full=True, references=own)
        values = rt.request(p['recovery_jobs'], p['query_model'], 'CURRENT_STUDENT_RECOVERY6', full=True, query=True)
        queries = [r for a, r in values]
        old = {r['job_id']: r for r in p['query_parent_rows']}
        for a, r in values:
            previous = E.N.S.arrays(old[r['job_id']]['path'])
            assert np.array_equal(a['actions'][:20], previous['actions'][:20])
            assert np.array_equal(a['native_counts'][:20], previous['native_counts'][:20])
            assert digest(a['initial_native_integration']) == old[r['job_id']]['array_hashes']['initial_native_integration']
        rt.request(p['recovery_jobs'], p['query_model'], 'CURRENT_STUDENT_RECOVERY6_EXACT', full=True, references=queries, query=True)
        gate = dict(all6_safe=all(r['success'] and not r['collision'] for r in queries),
                    teacher_results=R.M.tally(queries), parent5_historical_exact=True)
        save(OUT/'coverage_gate.json', gate)
        save(OUT/'coverage_rows.json', dict(own=own, queries=queries))
        if gate['all6_safe']:
            obj, shadow = make(p), make(p); initial = state(*obj)
            train = data(p, own, queries); trace = []
            for k in range(128):
                rt.role = 'ACTUAL_HISTORY_COVERAGE_BC'; rt.guard(True)
                va = []
                for which, objects in [('live', obj), ('shadow', shadow)]:
                    M.events('coverage_BC', which)('BEFORE', k)
                    va.append(step(objects, train))
                    M.events('coverage_BC', which)('AFTER', k)
                assert va[0] == va[1] and digest(state(*obj)) == digest(state(*shadow))
                completed += 1; trace.append(va[0])
                if k % 16 == 0:
                    save(OUT/'fit_progress.json', dict(steps=completed, latest=va[0], epoch=time.time()))
            save(OUT/'repair_trace.json', trace)
            proposal = export(obj, OUT/'proposal.pt.gz', p['model_bytes'])
            candidate, _ = M.evaluate(rt, p, proposal, 'COVERAGE_FINAL329')
            candidate['preserves_initial328'] = set(p['initial_success_ids']) <= set(candidate['success_ids'])
            candidate['repairs00'] = p['failed_job']['id'] in candidate['success_ids']
            candidate['development_admitted'] = len(candidate['success_ids']) == 329
        status = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj, initial); load(shadow, initial)
            assert digest(state(*obj)) == digest(initial)
        save(OUT/'failure.json', dict(error=error, partial_raw_preserved=True, no_restart=True))
    finally:
        if rt.pool: rt.pool.close(); rt.commands += 8
        ev = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines())) if (OUT/'optimizer_calls.jsonl').exists() else []
        counts = {w:sum(r[1:3] == [w, 'AFTER'] for r in ev) for w in ['live', 'shadow']}
        save(OUT/'summary.json', dict(state=status, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            candidate=candidate, proposal=proposal, gate=gate, own_rows=own, query_rows=queries,
            Actor_BC_live=counts['live'], Actor_BC_shadow=counts['shadow'], completed_paired_BC=completed,
            PPO_Actor_live=0, PPO_Value_live=0, new_native_RL_training=0,
            retained_model=p['research_initial'], research_proposal_not_promoted=True,
            fresh16=None, reload78=None, six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=status, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN); rt.lock.close()
    assert status == 'COMPLETE', error
    print('T460_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json'); verify(p, True); s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert s['physics'] == sum(r['physics'] for r in rows+replays)
    assert s['commands'] == 24+16+len(rows)+len(replays)
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert {k:digest(v) for k,v in a.items()} == {k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum()) == r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert [bool(v) for v in a['end_flags'][-1]] == [r['collision'], r['success'], r['timeout']]
        assert not r['administrative_censor'] and np.abs(a['actions']).max() <= 1
        if r['full']:
            assert np.array_equal(a['states'][1:], a['next_states'][:-1])
            assert digest(a) == r['array_digest']
            P.frozen = frozen
            try:
                M.audit_full_history(p, r, a)
            finally:
                P.frozen = H.ORIGINAL_FROZEN
            with torch.no_grad():
                expert = E.safe_initial(E.N.science(p), p).eval().requires_grad_(False) if r['teacher'] else None
                for k,x in enumerate(a['states']):
                    if expert is not None:
                        label = expert.base_actions(torch.tensor(x[None])).float()[0].numpy()
                        assert np.array_equal(label, a['teacher_actions'][k])
                assert digest(a['environment_rewards']) == digest(a['rewards'])
    for r in replays:
        original = next(x for x in rows if x['path'] == r['reference_path'])
        assert r['array_digest'] == original['array_digest'] and r['integration_sha256'] == original['integration_sha256']
        assert r['array_hashes'] == original['array_hashes'] and r['exact']
    calls = 0
    if s['gate']['all6_safe']:
        obj = make(p); train = data(p, s['own_rows'], s['query_rows'])
        trace = [step(obj, train) for _ in range(128)]; calls = 128
        assert trace == read(OUT/'repair_trace.json') and digest(state(*obj)) == s['proposal']['joint_hash']
        assert H.base_digest(obj[0]) == p['base_actor_digest']
        ev = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines()))
        assert len(ev) == 512
        for w in ['live', 'shadow']:
            assert [r[2:] for r in ev if r[:2] == ['coverage_BC', w]] == [[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
        final = [r for r in rows if r['role'] == 'COVERAGE_FINAL329']
        assert len(final) == len({r['job_id'] for r in final}) == 329
        assert R.M.tally(final[:54]) == s['candidate']['wide54'] and R.M.tally(final[54:]) == s['candidate']['old275']
        assert all(r['policy_hash'] == obj[0].phash() and r['loaded_model_sha256'] == s['proposal']['sha256'] for r in final)
    else:
        assert s['Actor_BC_live'] == s['Actor_BC_shadow'] == 0 and s['candidate'] is None
    assert len(s['own_rows']) == 5 and len(s['query_rows']) == 6 and len(replays) == 11
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), discarded_historical_residual_Adam=calls,
        complete_full120_hidden_and_teacher_labels_verified=True, eleven_exact_replays_verified=True,
        five_original_parent_policy_job_repeats_NOT_independent_new_success=True,
        final329_compact_NOT_full120_reconstructed=True, physics=s['physics'], commands=s['commands'],
        frozen_original_base_exact=True, new_native=0, new_PPO=0,
        fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T460_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T460',
            '--why=Bounded actual-history Stage coverage pilot','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),
            'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],
        all_stop=now+p['all_seconds'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T460_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'restore': C.restore(sys.argv[2])
        else: globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists(): save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
