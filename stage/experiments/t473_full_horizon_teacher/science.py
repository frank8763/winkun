"""One bounded full-remaining-horizon native recovery teacher, Stage only.

The nominal rule is fixed before CEM: use the complete original safeT255 old15
action sequence on its original 100-native-step decision clock. At the actual
A325 handoff (2000 native steps), index20 is the first nominal action. Linear
interpolation supplies fractional phase indices; indices beyond the successful
prior's end repeat its last action. No geometry, extra takeover time, seed grid,
or shortened terminal horizon is introduced. Zero theta is exactly this nominal.

Controls/candidates copy the two pure T391 functions with a registered lexical
terminal objective. Every proposal runs to true GOAL/CONTACT or the original
23334-step cap. A first safe GOAL may stop the finite search. A second native
pass regenerates the same adaptive search, then independently repeats its actual
selected candidate. No Actor, optimizer, or deployment planner is trained here.
"""
from pathlib import Path
import contextlib, copy, fcntl, importlib.util, json, os, shutil, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t473_full_horizon_teacher'
DOC = ROOT/'stage/docs/overnight_20261110'
spec = importlib.util.spec_from_file_location(
    'stage473_pinned471', ROOT/'stage/experiments/t471_reset_recovery/run.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
A, N = V.A, V.N
E, M, P, R, H, torch, np = V.E, V.M, V.P, V.R, V.H, V.torch, V.np
read, save, sha, digest, frozen = V.read, V.save, V.sha, V.digest, V.frozen
FIRST, CAP, HORIZON = 20, 23334, 214


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert p['out'] == str(OUT) and p['job']['id'] == 'dev_task6_15'
    assert p['job']['case']['step_cap'] == CAP
    assert p['teacher_first'] == p['prefix_holds'] == FIRST
    assert p['remaining_holds'] == HORIZON
    assert p['generations'] == 3 and p['population'] == 12 and p['elite'] == 3
    assert p['knots'] == 16 and p['alpha'] == .5 and p['seed'] == 20261110473
    assert p['initial_sd'] == [.12, 1.] and p['minimum_sd'] == [.03, .2]
    assert p['max_bytes'] == 32*2**20 and p['max_physics'] == 1600000
    assert p['max_commands'] == 80 and p['max_main_commands'] == 77
    assert p['preflight_commands'] == 2 and p['full_bytes'] == 448*1024
    assert p['main_seconds'] == 3000 and p['learning_seconds'] == 1
    assert p['audit_seconds'] == 900 and p['all_seconds'] == 5100
    assert M.used(OUT) <= p['max_bytes']
    assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for path, hsh in (p['inputs'] if full else p['protected_files']).items():
        assert sha(path) == hsh, path
    assert sha(p['learning_initial']['path']) == p['learning_initial']['sha256']
    assert sha(p['prior_controls_path']) == p['prior_controls_sha256']
    row = p['source_row']
    assert row['job_id'] == p['job']['id'] and row['full'] and not row['teacher']
    assert row['policy_hash'] == p['learning_initial']['policy_hash']
    assert row['loaded_model_sha256'] == p['learning_initial']['sha256']
    assert row['request']['model'] == p['learning_initial']['path']
    expected = dict(row['request']['job']); expected.pop('teacher_first', None)
    job = dict(p['job']); job.pop('teacher_first', None)
    assert job == expected and sha(row['path']) == row['sha256']
    assert not row['success'] and row['collision'] and not row['administrative_censor']
    if full:
        prior = read(ROOT/'stage/runs/t469_counterexample_recovery/protocol.json')
        summary = read(ROOT/'stage/runs/t469_counterexample_recovery/summary.json')
        assert row in summary['student_rows'] and p['learning_initial'] == prior['learning_initial']
        assert read(ROOT/'stage/runs/t469_counterexample_recovery/audit.json')['state'] == 'PASS'
        assert read(ROOT/'stage/runs/t471_reset_recovery/audit.json')['state'] == 'PASS'
        for key in ['absolute_native_stop', 'absolute_all_stop']:
            assert p[key] == prior[key]


def readonly_cpu():
    E.N.offline(); R.C.ban_learning()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)


def student(p):
    q = frozen(p, p['learning_initial']['path']).eval().requires_grad_(False)
    q.sample = False; q.rng = None; q.reset_noise()
    assert q.phash() == p['learning_initial']['policy_hash']
    assert N.base_digest(q) == p['base_actor_digest']
    assert all(not v.requires_grad and v.device.type == 'cpu' for v in q.parameters())
    return q


def controls(theta, reference, p):
    """T391 teacher.controls, unchanged physical-clock parameterization."""
    h = p['remaining_holds']; axis = np.arange(h); knots = np.linspace(0, h-1, 16)
    theta = np.asarray(theta, np.float64)
    assert theta.shape == (16, 2) and np.isfinite(theta).all()
    v = np.stack([np.interp(axis, knots, theta[:, j]) for j in range(2)], 1)
    clock = p['prefix_holds']+axis
    phase = np.maximum.accumulate(clock+5*np.clip(v[:, 1], -4., 4.))
    phase = np.clip(phase, 0, len(reference)-1)
    u = np.stack([np.interp(phase, np.arange(len(reference)), reference[:, j]) for j in range(2)], 1)
    u[:, 0] += v[:, 0]
    return np.clip(u, -1, 1).astype(np.float32)


def candidates(mean, sd, best, it, p):
    """T391 teacher.candidates: mean, best/zero, five antithetic pairs."""
    z = np.random.default_rng(np.random.SeedSequence([p['seed'], it])).normal(size=(5, 16, 2))
    result = [mean.copy(), np.zeros((16, 2)) if best is None else best.copy()]
    for row in z: result.extend([mean+row*sd, mean-row*sd])
    return np.asarray(result)


def prior_actions(p):
    a = E.N.S.arrays(p['prior_controls_path'])
    ref = a['actions']
    assert ref.ndim == 2 and ref.shape[1] == 2 and len(ref) > FIRST
    assert np.isfinite(ref).all() and np.abs(ref).max() <= 1
    assert a['end_flags'][-1, 1] and not a['end_flags'][:, 0].any()
    assert np.all(a['native_counts'][:-1] == 100)
    assert int(a['native_counts'][:FIRST].sum()) == FIRST*100
    nominal = controls(np.zeros((16, 2)), ref, p)
    assert np.array_equal(nominal, ref[np.minimum(FIRST+np.arange(HORIZON), len(ref)-1)].astype(np.float32))
    return ref, nominal


def score(row, actions, suffix_native):
    smooth = float(np.square(np.diff(actions.astype(np.float64), axis=0)).sum())
    if row['success'] and not row['collision']: return [0, float(suffix_native), smooth]
    return [2 if row['collision'] else 1, float(row['remaining_m']), smooth]


def payload(s):
    a = E.payload(s)
    # These are observed executed controls. No fictional teacher latent/raw
    # locations are supplied; any later BC adapter requires a new protocol.
    a['teacher_actions'] = a['actions'].copy()
    return a


def hashes(a): return {key: digest(value) for key, value in a.items()}


def actor_internal(s):
    return (s.q.previous_raw, s.q.previous_location, s.q.tape)


def recorded_actor_internal(a):
    keys = ['innovations', 'raw_actions', 'atom', 'locations', 'logp', 'stochastic']
    tape = [{k:np.asarray(a[k][i]).copy() for k in keys} for i in range(len(a['states']))]
    return (torch.tensor(a['raw_actions'][-1:]), torch.tensor(a['locations'][-1:]), tape)


def snapshot_hash(s):
    return digest(dict(integration=E.N.K.R.integration(s.env), packet=s.packet(),
        history=s.w.get(s.env), hidden=s.hidden, steps=s.steps,
        previous=s.previous, pn=s.pn, goal=s.goal, scene=s.scene, cap=s.cap,
        actor_internal=actor_internal(s)))


def recorded_snapshot_hash(a, p):
    return digest(dict(integration=a['final_native_integration'], packet=a['next_states'][-1],
        history=(a['history_poses'][-1], a['history_goals'][-1], a['history_times'][-1]),
        hidden=torch.tensor(a['last_hidden_after']).reshape(1, 1, 64),
        steps=int(a['native_counts'].sum()), previous=a['actions'][-1],
        pn=int(a['native_counts'][-1]), goal=a['task_goal_xy'],
        scene=a['scene_sha256'].tobytes().hex(), cap=p['job']['case']['step_cap'],
        actor_internal=recorded_actor_internal(a)))


def prefix_check(a, source, prefix_only=False):
    for key in ['states', 'next_states', 'poses', 'next_poses', 'actions', 'native_counts',
                'end_flags', 'rewards', 'environment_rewards', 'hidden_before', 'ownership',
                'raw_actions', 'locations', 'innovations', 'atom', 'logp', 'stochastic',
                'future_locations', 'hold_sha256']:
        if key in source: assert np.array_equal(a[key][:FIRST], source[key][:FIRST]), key
    for key in ['history_poses', 'history_goals', 'history_times']:
        assert np.array_equal(a[key][:FIRST+1], source[key][:FIRST+1]), key
    assert np.array_equal(a['initial_native_integration'], source['initial_native_integration'])
    assert np.array_equal(a['task_goal_xy'], source['task_goal_xy'])
    assert np.array_equal(a['scene_sha256'], source['scene_sha256'])
    assert int(a['native_counts'][:FIRST].sum()) == 2000
    if prefix_only:
        assert len(a['states']) == FIRST
        assert np.array_equal(a['next_states'][-1], source['states'][FIRST])
        assert np.array_equal(a['last_hidden_after'], source['hidden_before'][FIRST])
    else:
        for key in ['states', 'poses', 'hidden_before']:
            assert np.array_equal(a[key][:FIRST+1], source[key][:FIRST+1]), key


def write_full(path, a, p):
    assert not Path(path).exists(), path
    E.N.write_npz(path, E.N.S.packed(a), p['full_bytes'])
    assert Path(path).stat().st_size <= p['full_bytes']
    b = E.N.S.arrays(path)
    assert set(b) == set(a) and all(np.array_equal(a[k], b[k]) for k in a)


def setup(p, folder):
    E.set_out(OUT)
    module, native, create, witness, _ = E.N.S.P.C.B.setup(folder, E.N.science(p))
    module.ACTIVE = copy.deepcopy(module.ACTIVE)
    module.ACTIVE['circle_shift_x_m'] = p['job'].get('circle_shift_x_m', module.ACTIVE['circle_shift_x_m'])
    return module, native, create, witness


def session(p, native, create, witness, guard, charge):
    q = student(p)
    s = E.L.Session(p['job'], native, create, witness, guard, charge, q)
    s.hidden = torch.zeros((1, 1, 64), dtype=torch.float64)
    E.L.install_stop_handlers()
    return s


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def append(path, value):
    with Path(path).open('a') as f: f.write(json.dumps(value, allow_nan=False)+'\n')


def preflight():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert not (OUT/'preflight.json').exists()
    readonly_cpu(); reference, nominal = prior_actions(p)
    source = A.arrays(p['source_row']); q = student(p)
    # CPU-only recurrent audit uses the already recorded original failure.
    P.frozen = frozen
    try: M.audit_full_history(p, p['source_row'], source)
    finally: P.frozen = H.ORIGINAL_FROZEN
    mean = np.zeros((16, 2)); sd = np.broadcast_to(p['initial_sd'], (16, 2)).copy()
    proposal = candidates(mean, sd, None, 0, p)
    assert proposal.shape == (12, 16, 2) and np.array_equal(proposal[0], mean)
    assert np.array_equal(proposal, candidates(mean, sd, None, 0, p))
    assert np.array_equal(controls(proposal[0], reference, p), nominal)
    for theta in proposal:
        u = controls(theta, reference, p)
        assert u.shape == (HORIZON, 2) and np.isfinite(u).all() and np.abs(u).max() <= 1
    assert score(dict(success=True, collision=False), nominal, 21334) < score(dict(success=False, collision=False, remaining_m=.001), nominal, 1)
    assert score(dict(success=False, collision=False, remaining_m=99), nominal, 21334) < score(dict(success=False, collision=True, remaining_m=0), nominal, 1)
    folder = OUT/'preflight_native'; folder.mkdir()
    s = witness = None
    with (folder/'console.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            _, native, create, witness = setup(p, folder/'native')
            def cancel(physical=False):
                if physical: raise E.L.StageStop('T473 zero-native cancellation')
            def refuse(): raise AssertionError('Preflight cannot advance physics')
            s = session(p, native, create, witness, cancel, refuse)
            assert np.array_equal(s.initial, source['initial_native_integration'])
            assert np.array_equal(s.packet(), source['states'][0])
            snap = s.capture(); before = snapshot_hash(s); s.restore(snap)
            assert snapshot_hash(s) == before and s.steps == 0
            try: s.step(); raise AssertionError('Cancellation missing')
            except E.L.StageStop: pass
            assert s.steps == 0 and not s.rows
        finally:
            if s: s.close()
            if witness: witness.close()
    save(OUT/'preflight.json', dict(state='PASS', pid=os.getpid(), native=0, commands=2,
        optimizer=0, Actor_Adam=0, Value_Adam=0, backward=0,
        nominal_rule='prior complete action index20+axis, linear phase interpolation, last-action padding',
        nominal_sha256=digest(nominal), source_full_history_CPU_exact=True,
        session_original_geometry_clock_goal_history_hidden_capture_restore_exact=True,
        cancelled_before_first_native=True, synthetic_CEM_no_optimizer=True,
        first_candidate_zero_theta_is_nominal=True, all_sessions_closed=True))
    print('T473_PREFLIGHT_PASS_0NATIVE_0OPTIMIZER', flush=True)


def make_row(s, a, p, pass_index, index, kind, actual_physics, path):
    row = s.result()
    row.update(pass_index=pass_index, index=index, kind=kind,
        job_id=p['job']['id'], path=str(path), full=True, raw_teacher_control=True,
        teacher=True, teacher_first=FIRST, teacher_kind='native_full_horizon_phasewarp_CEM',
        teacher_source_SHA256=p['prior_controls_sha256'], teacher_hidden_burns_actual_history_from_h0=False,
        policy_hash=p['learning_initial']['policy_hash'], loaded_model_sha256=p['learning_initial']['sha256'],
        seed=None, request=dict(model=p['learning_initial']['path'], job=p['job'], seed=None),
        physics=int(a['native_counts'].sum()), actual_charged_physics=actual_physics,
        length=len(a['actions'])-FIRST, terminal_native=s.steps,
        score=score(row, a['actions'][FIRST:], actual_physics), array_digest=digest(a),
        array_hashes=hashes(a), integration_sha256=digest(a['final_native_integration']),
        end_flags=[bool(v) for v in a['end_flags'][-1]],
        actual_controls_digest=digest(a['actions'][FIRST:]),
        independent_new_sample=False,
        first_search_pass=pass_index == 0, new_native_control_query=kind == 'candidate',
        not_autonomous_Actor_success=True, raw_labels_require_future_explicit_adapter=True)
    return row


def main():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    readonly_cpu(); E.L.install_stop_handlers()
    reference, nominal = prior_actions(p); source = A.arrays(p['source_row'])
    start = time.time(); deadline = min(start+p['main_seconds'], p['absolute_native_stop'])
    physics = commands = 0; last = 0.; state = 'FAILED'; phase = 'SETUP'
    s = witness = None; records = []; passes = []; snapshots = []; selected_rows = []
    compact = dict(coefficients=np.zeros((36, 16, 2)), controls=np.zeros((36, HORIZON, 2), np.float32),
        scores=np.full((36, 3), np.nan), lengths=np.zeros(36, np.uint16),
        physics=np.zeros(36, np.int32), final_integration=np.zeros((36, 203)),
        end_flags=np.zeros((36, 3), np.uint8))
    lock = (ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT/'main_started.json', dict(pid=os.getpid(), epoch=start, deadline=deadline, optimizer=0))
    (OUT/'branches').mkdir()
    def guard(physical=False):
        nonlocal last
        E.L.check_stop()
        assert time.time() < deadline-5 and physics <= p['max_physics'] and commands <= p['max_main_commands']
        if not physical and time.time()-last > 3:
            last = time.time(); verify(p)
            save(OUT/'status.json', dict(state=phase, pid=os.getpid(), epoch=last,
                physics=physics, main_commands=commands, commands=commands+2, optimizer=0))
    def charge():
        nonlocal physics
        physics += 1
        assert physics <= p['max_physics']
    def begin(role):
        nonlocal commands, phase
        phase = role; guard(); commands += 1; guard()
        return physics
    def end(before, **extra):
        append(OUT/'commands.jsonl', dict(id=commands, role=phase,
            physics=physics-before, cumulative_physics=physics, pid=os.getpid(), **extra))
    def branch(pass_index, index, u, snap, kind):
        before = begin(f'PASS{pass_index}_{kind.upper()}_{index:02d}')
        s.restore(snap); assert snapshot_hash(s) == snapshots[pass_index]['sha256']
        for action in u:
            s.step(action, owner=1, actor_active=True)
            if s.done(): break
        assert s.done() and (s.steps == CAP or bool(s.info.get('reach_goal')) or bool(s.info.get('collision')))
        a = payload(s); prefix_check(a, source)
        assert len(a['actions']) <= FIRST+HORIZON and physics-before == int(a['native_counts'][FIRST:].sum())
        assert np.array_equal(a['actions'][FIRST:], u[:len(a['actions'])-FIRST])
        assert not a['ownership'][:FIRST].any() and a['ownership'][FIRST:].all()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert s.q.phash() == p['learning_initial']['policy_hash'] and not s.q.sample
        path = OUT/'branches'/f'{index:02d}.npz' if kind == 'candidate' else OUT/f'selected_pass{pass_index}.npz'
        row = make_row(s, a, p, pass_index, index, kind, physics-before, path)
        row['planned_controls_digest'] = digest(u)
        # The prescribed mean/zero and later best/mean proposals can duplicate
        # controls. They still execute and are charged, never new independent
        # control sequences merely because they have another candidate ID.
        row['independent_new_sample'] = False
        if kind == 'candidate':
            if pass_index == 0:
                write_full(path, a, p); row['sha256'] = sha(path)
            else:
                old = next(v for v in records if v['pass_index'] == 0 and v['index'] == index)
                b = E.N.S.arrays(old['path'])
                assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a), ('Native repeat differs', index)
                row['sha256'] = sha(old['path'])
                ignore = {'pass_index', 'independent_new_sample'}
                assert {k:v for k,v in row.items() if k not in ignore} == {k:v for k,v in old.items() if k not in ignore}
                row['exact_replay_not_independent'] = True
            records.append(row); append(OUT/'branches.jsonl', row)
        else:
            best = next(v for v in records if v['pass_index'] == pass_index and v['index'] == index)
            assert row['array_hashes'] == best['array_hashes'] and row['array_digest'] == best['array_digest']
            assert row['score'] == best['score'] and row['integration_sha256'] == best['integration_sha256']
            write_full(path, a, p); row['sha256'] = sha(path)
            if pass_index:
                b = E.N.S.arrays(selected_rows[0]['path'])
                assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
                assert row['array_hashes'] == selected_rows[0]['array_hashes']
            selected_rows.append(row); save(OUT/'selected.json', selected_rows)
        end(before, pass_index=pass_index, index=index, kind=kind, array_digest=row['array_digest'])
        return row
    try:
        with (OUT/'native_console.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            before = begin('NATIVE_SETUP')
            _, native, create, witness = setup(p, OUT/'native'); end(before)
            for pass_index in range(2):
                before = begin(f'PASS{pass_index}_ORIGINAL_A325_PREFIX')
                s = session(p, native, create, witness, guard, charge)
                assert np.array_equal(s.initial, source['initial_native_integration'])
                for _ in range(FIRST): s.step(owner=0)
                assert s.steps == 2000 and not s.done()
                a = payload(s); prefix_check(a, source, True)
                path = OUT/f'prefix_pass{pass_index}.npz'; write_full(path, a, p)
                snap = s.capture(); hsh = snapshot_hash(s)
                assert hsh == recorded_snapshot_hash(a, p)
                item = dict(pass_index=pass_index, sha256=hsh, prefix_path=str(path), prefix_sha256=sha(path),
                    prefix_array_digest=digest(a), prefix_array_hashes=hashes(a), steps=s.steps,
                    packet_sha256=digest(s.packet()), integration_sha256=digest(a['final_native_integration']),
                    actor_internal_sha256=digest(actor_internal(s)))
                assert item['actor_internal_sha256'] == digest(recorded_actor_internal(a))
                if pass_index: assert item['sha256'] == snapshots[0]['sha256'] and item['prefix_array_digest'] == snapshots[0]['prefix_array_digest']
                snapshots.append(item); save(OUT/'snapshots.json', snapshots)
                end(before, pass_index=pass_index, snapshot_sha256=hsh)
                mean = np.zeros((16, 2)); sd = np.broadcast_to(p['initial_sd'], (16, 2)).copy()
                best = None; best_id = None; best_score = None; traces = []; stopped = False; queries = 0
                for generation in range(p['generations']):
                    values = candidates(mean, sd, best, generation, p); ids = []
                    for j, theta in enumerate(values):
                        index = generation*12+j; u = controls(theta, reference, p)
                        if pass_index == 0:
                            compact['coefficients'][index] = theta; compact['controls'][index] = u
                        else:
                            assert np.array_equal(theta, compact['coefficients'][index])
                            assert np.array_equal(u, compact['controls'][index])
                        row = branch(pass_index, index, u, snap, 'candidate')
                        if pass_index == 0:
                            compact['scores'][index] = row['score']; compact['lengths'][index] = row['length']
                            compact['physics'][index] = row['actual_charged_physics']
                            compact['final_integration'][index] = E.N.S.arrays(row['path'])['final_native_integration']
                            compact['end_flags'][index] = row['end_flags']
                        assert np.array_equal(compact['scores'][index], np.asarray(row['score']))
                        ids.append(index); queries += 1
                        if best_id is None or (row['score'], index) < (best_score, best_id):
                            best_id, best_score, best = index, row['score'], theta.copy()
                        stopped = row['success'] and not row['collision']
                        if stopped: break
                    elite = sorted(ids, key=lambda k:(compact['scores'][k].tolist(), k))[:p['elite']]
                    trace = dict(generation=generation, queries=ids, elite=elite,
                        mean_before=mean.tolist(), sd_before=sd.tolist(), best_id=best_id,
                        best_score=best_score, safe_GOAL=bool(stopped))
                    if not stopped:
                        assert len(elite) == 3
                        mean = .5*mean+.5*compact['coefficients'][elite].mean(0)
                        sd = np.maximum(np.asarray(p['minimum_sd']), .5*sd+.5*compact['coefficients'][elite].std(0))
                    trace.update(mean_after=mean.tolist(), sd_after=sd.tolist()); traces.append(trace)
                    save(OUT/f'cem_pass{pass_index}.json', traces)
                    if stopped: break
                if pass_index: assert traces == passes[0]['iterations'] and best_id == passes[0]['selected_index']
                selected = branch(pass_index, best_id, compact['controls'][best_id], snap, 'selected')
                passes.append(dict(pass_index=pass_index, iterations=traces, queries=queries,
                    selected_index=best_id, selected_array_digest=selected['array_digest'],
                    safe_GOAL=bool(selected['success'] and not selected['collision']),
                    snapshot_sha256=hsh, independent_search_pass=pass_index == 0))
                save(OUT/'passes.json', passes)
                E.N.write_npz(OUT/'search.npz', compact, 2*2**20)
                s.close(); s = None
            state = 'COMPLETE'
    except BaseException:
        save(OUT/'failure.json', dict(error=traceback.format_exc(), phase=phase,
            physics=physics, main_commands=commands, raw_preserved=True, no_restart=True))
        if s is not None and s.rows:
            try: write_full(OUT/'partial.npz', payload(s), p)
            except BaseException: save(OUT/'partial_save_error.json', dict(error=traceback.format_exc()))
    finally:
        if s: s.close()
        if witness: witness.close()
        result = selected_rows[0] if selected_rows else None
        valid = bool(state == 'COMPLETE' and result and result['success'] and not result['collision'])
        unique_controls = len({r['planned_controls_digest'] for r in records if r['pass_index'] == 0})
        save(OUT/'summary.json', dict(state=state, pid=os.getpid(), elapsed_s=time.time()-start,
            physics=physics, main_commands=commands, preflight_commands=2, commands=commands+2,
            optimizer=0, optimizer_steps=0, Actor_updates=0, Actor_Adam=0, Value_Adam=0, backward=0,
            learning_initial=p['learning_initial'], frozen_model_sha256=sha(p['learning_initial']['path']),
            source_row=p['source_row'], prior_controls_sha256=p['prior_controls_sha256'],
            nominal_array_digest=digest(nominal), passes=passes, selected_rows=selected_rows,
            result=result, valid_recovery_label=valid, all_safe=valid, teacher_first=FIRST,
            control_search_only=True, raw_controls_NOT_neural_teacher_latent_labels=True,
            first_pass_executed_candidates=sum(r['pass_index'] == 0 for r in records),
            first_pass_unique_planned_control_sequences=unique_controls,
            duplicate_control_candidates_charged_not_independent=True,
            future_learning_requires_explicit_adapter=True, exact_replays_NOT_independent=True,
            reused_sources_NOT_new_independent_samples=True, all_sessions_closed=True,
            full_329=None, fresh16=None, reload78=None, six_gate=False, eight_started=False))
        save(OUT/'status.json', dict(state=state, pid=os.getpid(), epoch=time.time(),
            physics=physics, main_commands=commands, commands=commands+2, optimizer=0))
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    assert state == 'COMPLETE', 'T473 engineering failure; evidence preserved, no automatic restart'
    verify(p, True)
    print('T473_MAIN_COMPLETE', valid, physics, commands+2, flush=True)


def audit_payload(p, row, a, source, guard):
    guard(); assert hashes(a) == row['array_hashes'] and digest(a) == row['array_digest']
    assert digest(a['final_native_integration']) == row['integration_sha256']
    assert a['states'].dtype == np.float32 and a['states'].shape[1] == 120
    assert a['hidden_before'].shape == (len(a['states']), 64)
    for v in a.values():
        if np.issubdtype(v.dtype, np.number): assert np.isfinite(v).all()
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    counts = a['native_counts'].astype(np.int64); elapsed = np.r_[0, np.cumsum(counts)]
    assert np.all((counts > 0) & (counts <= 100)) and int(counts.sum()) == row['physics'] <= CAP
    assert np.max(np.abs(a['states'][:, 112]-(CAP-elapsed[:-1])/CAP)) < 2e-6
    assert np.max(np.abs(a['next_states'][:, 112]-(CAP-elapsed[1:])/CAP)) < 2e-6
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
    assert [bool(v) for v in a['end_flags'][-1]] == row['end_flags'] == [row['collision'], row['success'], row['timeout']]
    if row['timeout']: assert row['physics'] == CAP and int(counts[-1]) == 34
    assert not row['administrative_censor'] and np.array_equal(a['rewards'], a['environment_rewards'])
    assert np.isfinite(a['actions']).all() and np.abs(a['actions']).max() <= 1
    assert np.array_equal(a['teacher_actions'], a['actions'])
    assert not a['ownership'][:FIRST].any() and a['ownership'][FIRST:].all() and not a['stochastic'].any()
    prefix_check(a, source)
    assert abs(float(np.linalg.norm(a['next_poses'][-1, :2]-a['task_goal_xy']))-row['remaining_m']) < 1e-12
    assert row['actual_charged_physics'] == int(counts[FIRST:].sum()) == row['physics']-2000
    assert row['score'] == score(row, a['actions'][FIRST:], row['actual_charged_physics'])
    # Pinned audit independently rebuilds all120 observations, recurrent64
    # carry, proposal density/atoms/raw/location, ownership and native rewards.
    P.frozen = frozen
    try: M.audit_full_history(dict(p, teacher_first=FIRST), row, a)
    finally: P.frozen = H.ORIGINAL_FROZEN
    guard()


def audit():
    p = read(OUT/'protocol.json'); verify(p, True)
    summary = read(OUT/'summary.json')
    assert summary['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    assert not Path('/proc', str(summary['pid'])).exists(), 'Native main must have exited before independent audit'
    readonly_cpu(); start = time.time()
    deadline = min(start+p['audit_seconds'], p['absolute_all_stop'])
    def guard(): assert time.time() < deadline-5
    source = A.arrays(p['source_row']); reference, nominal = prior_actions(p)
    assert digest(nominal) == summary['nominal_array_digest'] == read(OUT/'preflight.json')['nominal_sha256']
    student(p)
    P.frozen = frozen
    try: M.audit_full_history(p, p['source_row'], source)
    finally: P.frozen = H.ORIGINAL_FROZEN
    passes = read(OUT/'passes.json'); snapshots = read(OUT/'snapshots.json')
    rows = jsonl(OUT/'branches.jsonl'); commands = jsonl(OUT/'commands.jsonl')
    selected = read(OUT/'selected.json')
    assert passes == summary['passes'] and selected == summary['selected_rows']
    assert len(passes) == len(snapshots) == len(selected) == 2
    assert passes[0]['iterations'] == passes[1]['iterations']
    assert passes[0]['selected_index'] == passes[1]['selected_index']
    with np.load(OUT/'search.npz', allow_pickle=False) as z: data = {k:z[k].copy() for k in z.files}
    assert data['coefficients'].shape == (36, 16, 2) and data['controls'].shape == (36, HORIZON, 2)
    first_rows = [r for r in rows if r['pass_index'] == 0]
    second_rows = [r for r in rows if r['pass_index'] == 1]
    assert len(first_rows) == len(second_rows) == passes[0]['queries'] <= 36
    assert [r['index'] for r in first_rows] == [r['index'] for r in second_rows]
    assert len(list((OUT/'branches').glob('*.npz'))) == len(first_rows)
    byid = {r['index']:r for r in first_rows}
    for snapshot in snapshots:
        guard(); path = snapshot['prefix_path']; assert sha(path) == snapshot['prefix_sha256']
        a = E.N.S.arrays(path); prefix_check(a, source, True)
        assert hashes(a) == snapshot['prefix_array_hashes'] and digest(a) == snapshot['prefix_array_digest']
        assert recorded_snapshot_hash(a, p) == snapshot['sha256']
        assert digest(recorded_actor_internal(a)) == snapshot['actor_internal_sha256']
        assert digest(source['states'][FIRST]) == snapshot['packet_sha256']
        assert digest(a['final_native_integration']) == snapshot['integration_sha256']
    assert snapshots[0]['sha256'] == snapshots[1]['sha256']
    assert snapshots[0]['prefix_array_digest'] == snapshots[1]['prefix_array_digest']
    for row, repeat in zip(first_rows, second_rows):
        guard(); assert sha(row['path']) == row['sha256'] == repeat['sha256']
        ignore = {'pass_index', 'independent_new_sample', 'exact_replay_not_independent'}
        assert {k:v for k,v in row.items() if k not in ignore} == {k:v for k,v in repeat.items() if k not in ignore}
        assert repeat['exact_replay_not_independent'] and not repeat['independent_new_sample']
        a = E.N.S.arrays(row['path']); audit_payload(p, row, a, source, guard)
        i = row['index']; n = row['length']
        assert np.array_equal(a['actions'][FIRST:], data['controls'][i, :n])
        assert digest(data['controls'][i]) == row['planned_controls_digest']
        assert digest(a['actions'][FIRST:]) == row['actual_controls_digest']
        assert row['score'] == data['scores'][i].tolist() and n == int(data['lengths'][i])
        assert row['actual_charged_physics'] == int(data['physics'][i])
        assert np.array_equal(a['final_native_integration'], data['final_integration'][i])
        assert np.array_equal(a['end_flags'][-1], data['end_flags'][i])
    # Recompute random draws, all executed proposals, adaptive elites and first
    # success stopping independently from saved native outcomes. No physics.
    mean = np.zeros((16, 2)); sd = np.broadcast_to(p['initial_sd'], (16, 2)).copy()
    best = None; best_id = None; best_score = None; seen = []
    for generation, trace in enumerate(passes[0]['iterations']):
        guard(); values = candidates(mean, sd, best, generation, p); ids = []
        assert trace['generation'] == generation
        assert np.array_equal(mean, np.asarray(trace['mean_before'])) and np.array_equal(sd, np.asarray(trace['sd_before']))
        for j, theta in enumerate(values):
            i = generation*12+j
            if i not in byid: break
            row = byid[i]; ids.append(i); seen.append(i)
            assert np.array_equal(theta, data['coefficients'][i])
            assert np.array_equal(controls(theta, reference, p), data['controls'][i])
            if best_id is None or (row['score'], i) < (best_score, best_id):
                best_id, best_score, best = i, row['score'], theta.copy()
            if row['success'] and not row['collision']: break
        assert ids == trace['queries']
        stopped = bool(byid[ids[-1]]['success'] and not byid[ids[-1]]['collision'])
        assert stopped == trace['safe_GOAL'] and best_id == trace['best_id'] and best_score == trace['best_score']
        elite = sorted(ids, key=lambda i:(byid[i]['score'], i))[:3]
        assert elite == trace['elite']
        if not stopped:
            assert len(ids) == 12 and len(elite) == 3
            mean = .5*mean+.5*data['coefficients'][elite].mean(0)
            sd = np.maximum(np.asarray(p['minimum_sd']), .5*sd+.5*data['coefficients'][elite].std(0))
        else: assert generation == len(passes[0]['iterations'])-1
        assert np.array_equal(mean, np.asarray(trace['mean_after'])) and np.array_equal(sd, np.asarray(trace['sd_after']))
    assert seen == [r['index'] for r in first_rows]
    assert stopped or len(passes[0]['iterations']) == 3
    assert best_id == passes[0]['selected_index'] == min(first_rows, key=lambda r:(r['score'], r['index']))['index']
    selected_arrays = []
    for pass_index, row in enumerate(selected):
        assert row['pass_index'] == pass_index and row['index'] == best_id
        assert sha(row['path']) == row['sha256']
        a = E.N.S.arrays(row['path']); audit_payload(p, row, a, source, guard)
        assert row['array_hashes'] == byid[best_id]['array_hashes']
        assert row['array_digest'] == passes[pass_index]['selected_array_digest']
        selected_arrays.append(a)
    assert set(selected_arrays[0]) == set(selected_arrays[1])
    assert all(np.array_equal(selected_arrays[0][k], selected_arrays[1][k]) for k in selected_arrays[0])
    # Repeats and selected confirmations are fully charged, never independent
    # samples. Snapshot restoration itself executes no native step.
    expected_physics = 4000+sum(r['actual_charged_physics'] for r in rows+selected)
    expected_commands = 1+2*(2+len(first_rows))
    assert summary['physics'] == sum(r['physics'] for r in commands) == expected_physics <= p['max_physics']
    assert summary['main_commands'] == len(commands) == expected_commands <= p['max_main_commands']
    assert [r['id'] for r in commands] == list(range(1, expected_commands+1))
    assert summary['commands'] == expected_commands+2 <= p['max_commands']
    expected_roles = ['NATIVE_SETUP']
    for pass_index in range(2):
        expected_roles += [f'PASS{pass_index}_ORIGINAL_A325_PREFIX']
        expected_roles += [f'PASS{pass_index}_CANDIDATE_{i:02d}' for i in seen]
        expected_roles += [f'PASS{pass_index}_SELECTED_{best_id:02d}']
    assert [r['role'] for r in commands] == expected_roles
    cumulative = 0
    for command in commands:
        cumulative += command['physics']; assert cumulative == command['cumulative_physics']
        if command.get('kind') in ['candidate', 'selected']:
            candidate = byid[command['index']]
            assert command['physics'] == candidate['actual_charged_physics']
            assert command['array_digest'] == candidate['array_digest']
    assert len([r for r in commands if 'PREFIX' in r['role']]) == 2
    assert all(r['physics'] == 2000 for r in commands if 'PREFIX' in r['role'])
    assert commands[0]['role'] == 'NATIVE_SETUP' and commands[0]['physics'] == 0
    assert summary['result'] == selected[0]
    unique_controls = len({r['planned_controls_digest'] for r in first_rows})
    assert summary['first_pass_executed_candidates'] == len(first_rows)
    assert summary['first_pass_unique_planned_control_sequences'] == unique_controls
    valid = bool(selected[0]['success'] and not selected[0]['collision'])
    assert summary['valid_recovery_label'] == summary['all_safe'] == valid
    assert summary['frozen_model_sha256'] == sha(p['learning_initial']['path']) == p['learning_initial']['sha256']
    for k in ['optimizer', 'optimizer_steps', 'Actor_updates', 'Actor_Adam', 'Value_Adam', 'backward']: assert summary[k] == 0
    assert not summary['six_gate'] and not summary['eight_started'] and summary['all_sessions_closed']
    assert summary['full_329'] is None and summary['fresh16'] is None and summary['reload78'] is None
    assert not list(OUT.glob('partial*.npz'))
    verify(p, True); guard()
    save(OUT/'teacher_diagnosis.json', dict(state='COMPLETE_FIXED_TEACHER_DIAGNOSTIC',
        valid_recovery_label=valid, selected_index=best_id, selected=selected[0],
        source_actual_student_prefix_holds=FIRST, source_actual_student_prefix_native=2000,
        full_original_cap=CAP, full_remaining_native=CAP-2000, planned_remaining_holds=HORIZON,
        nominal_array_digest=digest(nominal), executed_first_pass_candidates=len(first_rows),
        unique_planned_control_sequences=unique_controls,
        duplicate_controls_still_charged=True, two_native_search_passes_exact=True,
        selected_independent_native_regenerations_exact=True, optimizer=0,
        no_Actor_learning_or_autonomous_success_claim=True,
        raw_controls_require_new_explicit_adapter_before_learning=True,
        six_gate=False, eight_started=False))
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), elapsed_s=time.time()-start,
        native=0, optimizer=0, backward=0, all_saved_full120_history_rewards_density_and_student64_carry_rebuilt=True,
        both_native_search_passes_exact_all_arrays_and_integrations=True,
        every_saved_candidate_array_hash_and_terminal_cost_exact=True,
        recorded_prefix_snapshot_hashes_independently_reconstructed=True,
        CEM_controls_random_draws_elites_stopping_and_selection_recomputed=True,
        selected_two_independent_native_regenerations_all_arrays_exact=True,
        hold_hashes_producer_native_two_pass_exact_not_offline_native_reconstruction=True,
        physics=summary['physics'], commands=summary['commands'], valid_recovery_label=valid,
        raw_control_labels_require_future_adapter=True, not_autonomous_Actor_success=True,
        exact_replays_not_independent=True, six_gate=False, eight_started=False))
    print('T473_INDEPENDENT_ZERO_NATIVE_AUDIT_PASS', valid, flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['preflight', 'main', 'audit']
    globals()[sys.argv[1]]()
