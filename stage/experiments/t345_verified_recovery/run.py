"""Fixed failing-prefix recovery availability, no learning or deployed routing."""
from pathlib import Path
import sys, copy, io, gzip, json, os, time, signal, shutil, traceback, contextlib, subprocess
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'stage/experiments/t344_neighbour_recovery'))
import t344_core as B
K, T, np, torch = B.K, B.T, B.np, B.torch
read, save, digest, arrays = B.read, B.save, B.digest, B.arrays
OUT = ROOT / 'stage/runs/t345_verified_recovery'
DOC = ROOT / 'stage/docs/review_20261104'
BASE_NAMES = ['states', 'actions', 'native_counts', 'next_states', 'poses', 'next_poses',
              'end_flags', 'rewards', 'history_poses', 'history_goals', 'history_times', 'environment_rewards']


def forbid(*args, **kwargs):
    raise AssertionError('T345 has zero optimizer updates')


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for f, h in p['input_hashes'].items():
        assert K.C.sha(f) == h, f
    assert K.S.used(ROOT / 'stage/runs') - K.S.used(OUT) + p['final_bytes'] <= p['global_bytes']
    assert K.S.used(OUT) <= p['final_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']


def actor(p, name):
    q = T.Actor(p, 'capg').eval().requires_grad_(False)
    v = torch.load(io.BytesIO(gzip.decompress(Path(p['models'][name]['path']).read_bytes())),
                   weights_only=False, map_location='cpu')
    q.load_state_dict(dict(q.state_dict(), **v['state']['actor']), strict=True)
    assert digest(q.state_dict()) == p['models'][name]['policy_hash']
    assert not q.memory_on
    return q


def prepare():
    assert not OUT.exists()
    close = read(DOC / 'T344_CLOSEOUT.json')
    assert close['additive_all_actual_updates_reconstruction_pass'] and not close['original_audit_pass']
    receipt = ROOT / 'stage/docs/long_distance_20261101/release_receipt_T344.json'
    assert read(receipt)['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p = copy.deepcopy(read(ROOT / 'stage/experiments/t344_neighbour_recovery/protocol.json'))
    rows = [json.loads(l) for l in (ROOT / 'stage/runs/t344_neighbour_recovery/cases.jsonl').read_text().splitlines()]
    sources = []
    for ident in ['training_r0_n0', 'training_r0_n2']:
        r = next(x for x in rows if x['id'] == ident)
        assert K.C.sha(r['path']) == r['sha256']
        a = arrays(r['path'])
        assert r['collision'] and not r['success'] and r['complete']
        # One fixed rule on consumed failures; no outcome-adaptive time search.
        count = (r['physics_steps'] - 5000) // 100
        assert count > 0 and np.all(a['native_counts'][:count] == 100)
        sources.append(dict(row=r, prefix_decisions=int(count), prefix_physics=int(count * 100),
                            seconds_before_historical_contact=(r['physics_steps'] - count * 100) * .002))
    p.update(experiment='T345_FIXED_FAILED_PREFIX_VERIFIED_RECOVERY_AVAILABILITY', sources=sources,
             models={
                 'T340': dict(path=p['student_initial_model'], policy_hash=p['student_initial_policy_hash']),
                 'T342': dict(path=p['supporting_model'], policy_hash=p['supporting_policy_hash'])},
             suffixes=['historical_actions', 'T340', 'T342'],
             geometry_commands=1, max_commands=13, max_physics=480000,
             wall_seconds=900, final_reserve_seconds=120, audit_seconds=300, all_work_seconds=2400,
             final_bytes=int(4.5 * 1024**2), episode_bytes=512 * 1024, partial_bytes=512 * 1024,
             metadata_bytes=768 * 1024, log_bytes=256 * 1024,
             native_step_cap=40000, optimizer_calls=0, new_training_trajectories=0,
             contract='Two consumed T344 r0 n0/n2 failures. One handoff at floor((historical collision physical step-5000)/100)*100 native; entire prefix genuinely executed with original actions. Frozen historical suffix vs T340 vs T342, each exact native replay,6primary6replay+1zero-physics geometry,13commands480k allphysics/900s/own4.5MiB/global1GiB/free40GiB. Preserve goal/history120/previous-action/actuator/clock/integration/no teleport or reset at handoff. Same original speed5/force120/wide1.17m/80s; no action/time/model/seed search. Mixed behaviour is diagnostic offline data,not onpolicy or autonomous-from-start success. Only safe terminal GOAL plus full replay and audit can supply a verified recovery suffix; no learned expert assumed. Zero Actor/Q/Adam; no original275/54/fresh16/reload78 or7/8; no planner/map/waypoint/deployment teacher/router/Phase. Stop after fixed cases; no automatic reattempt if allfail. T344 original audit staysfailed, additive reconstruction separate.')
    assert 6 * p['episode_bytes'] + p['partial_bytes'] + p['metadata_bytes'] + p['log_bytes'] <= p['final_bytes']
    files = set(p['input_hashes']) | {str(HERE / 'run.py'), str(DOC / 'T345_PLAN.md'), str(receipt),
                                    str(DOC / 'T344_CLOSEOUT.json'), str(DOC / 'T344_AUDIT_RECOVERY.json')}
    files |= {s['row']['path'] for s in sources} | {m['path'] for m in p['models'].values()}
    p['input_hashes'] = {f: K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT / 'stage/runs') + p['final_bytes'] <= p['global_bytes']
    OUT.mkdir(); (OUT / 'episodes').mkdir()
    save(HERE / 'protocol.json', p); save(OUT / 'protocol.json', p)
    print(json.dumps(dict(state='REGISTERED', native=0, Adam=0,
                         prefixes=[s['prefix_physics'] for s in sources],
                         max_commands=13, max_physics=480000,
                         reserved_global=K.S.used(ROOT / 'stage/runs') - K.S.used(OUT) + p['final_bytes'])))


def preflight():
    p = read(HERE / 'protocol.json'); verify(p)
    assert not (OUT / 'preflight.json').exists()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.optim.Adam.step = forbid
    folder = OUT / 'preflight'; folder.mkdir()
    with (folder / 'factory.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        m, native, create, w, reward = B.setup(folder / 'native', p)
        job = next(j for j in p['course_dev_jobs'] if j['id'] == p['sources'][0]['row']['job_id'])
        env, obs, _ = K.R.task_env(create, job)
        try:
            obs = K.C.perturb_anchor(env, obs, job)
            g = B.geometry(env, obs, w, 40000)
            ref = arrays(p['sources'][0]['row']['path'])
            assert digest(np.asarray(g['full120'], np.float32)) == digest(ref['states'][0])
            assert np.array_equal(K.R.integration(env), ref['initial_native_integration'])
            assert env.max_wheel == 5 and env.max_steps == 40000
            assert np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel, env.id_rr_wheel]], [[-120., 120.], [-120., 120.]])
            first = {}
            for name in p['models']:
                q = actor(p, name)
                with torch.no_grad():
                    x = torch.tensor(ref['states'][:1])
                    action = q.step(x)[0].numpy()
                    assert np.array_equal(action, q.draw(x, torch.zeros((1, 2), dtype=torch.float64), True)['actions'].numpy())
                first[name] = action[0].tolist()
            assert env.step_count == 0
            save(OUT / 'geometry.json', g)
            save(OUT / 'preflight.json', dict(state='PASS', geometry_commands=1, physics_steps=0, optimizer_calls=0,
                                             both_models_strict_load=True, native_initial_state_exact=True,
                                             first_actions=first, geometry_sha256=K.C.sha(OUT / 'geometry.json')))
        finally:
            w.events.pop(id(env), None); env.close(); w.close()
    print('T345 preflight PASS:1geometry/0native/0Adam,bothmodels actual action path')


def launch():
    p = read(HERE / 'protocol.json'); verify(p)
    assert read(OUT / 'preflight.json')['state'] == 'PASS'
    assert not (OUT / 'launch.json').exists() and not (OUT / 'status.json').exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip()
    r = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    assert r['root_tree'] == tree
    start = time.time()
    command = ['systemd-inhibit', '--what=sleep', '--mode=block', '--why=Bounded Stage T345 recovery diagnostic',
               'timeout', '--signal=TERM', '--kill-after=1s', str(p['wall_seconds']) + 's',
               'taskset', '-c', '0', str(ROOT / 'run_stage.sh'), str(HERE / 'run.py'), 'run']
    with (OUT / 'main.log').open('wb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT / 'launch.json', dict(pid=proc.pid, launch_epoch=start, deadline_epoch=start + p['wall_seconds'],
                                  all_work_deadline_epoch=start + p['all_work_seconds'], command=command,
                                  source_remote_commit=r['remote_commit'], source_root_tree=tree,
                                  protocol_sha256=K.C.sha(HERE / 'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED', pid=proc.pid, deadline_epoch=start + p['wall_seconds'])))


def prefix_exact(a, source, n):
    for name in BASE_NAMES:
        stop = n + 1 if name.startswith('history_') else n
        assert np.array_equal(a[name][:stop], source[name][:stop]), name


def main():
    p = read(HERE / 'protocol.json'); verify(p)
    assert read(OUT / 'preflight.json')['state'] == 'PASS' and not (OUT / 'status.json').exists()
    assert K.C.sha(OUT / 'geometry.json') == read(OUT / 'preflight.json')['geometry_sha256']
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.optim.Adam.step = forbid
    start = time.time(); deadline = min(start + p['wall_seconds'], read(OUT / 'launch.json')['deadline_epoch'])
    physics = 0; commands = 1; rows = []; reps = []; state = 'FAILED'; phase = 'STARTING'; last = 0.; s = None; w = None
    def status(force=False):
        nonlocal last
        if not force and time.time() - last < 2: return
        last = time.time()
        save(OUT / 'status.json', dict(state=phase, pid=os.getpid(), main_epoch=start, updated_epoch=last,
                                      deadline_epoch=deadline, commands=commands, physics_steps=physics, cases=len(rows), replays=len(reps)))
    def guard():
        assert time.time() < deadline - 10 and physics < p['max_physics']
        if time.time() - last >= 2:
            assert K.S.used(OUT) <= p['final_bytes'] and K.S.used(ROOT / 'stage/runs') <= p['global_bytes']
            assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']; status()
    def physical():
        nonlocal physics
        physics += 1
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T345 bounded deadline')))
    signal.alarm(max(1, int(deadline - time.time()) - 2))
    try:
        models = {name: actor(p, name) for name in p['models']}
        with (OUT / 'factory.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            m, native, create, w, reward = B.setup(OUT / 'native', p)
            geo = read(OUT / 'geometry.json')
            for si, item in enumerate(p['sources']):
                source = arrays(item['row']['path']); n = item['prefix_decisions']; handoffs = []
                job = next(j for j in p['course_dev_jobs'] if j['id'] == item['row']['job_id'])
                for name in p['suffixes']:
                    reference = None
                    for replica in ['primary', 'exact_native_replay']:
                        assert time.time() + p['final_reserve_seconds'] + 60 < deadline and commands < p['max_commands']
                        assert physics + 40000 <= p['max_physics']; commands += 1
                        phase = f'source{si}_{name}_{replica}'; status(True)
                        cp = copy.deepcopy(read(p['course_protocol'])); cp.update(job=job, initial_geometry=geo)
                        s = m.R.Session(cp, native, create, w, reward, guard, physical)
                        for k in range(n): s.step(source['actions'][k], int(source['native_counts'][k]))
                        prefix_exact(s.arrays(), source, n)
                        assert not (s.term or s.trunc) and s.steps == item['prefix_physics']
                        handoff = K.R.integration(s.env).copy()
                        x = K.T.L.packet(s.env, s.obs, s.steps, 40000, s.previous, s.pn)
                        assert np.array_equal(x, source['states'][n])
                        if handoffs: assert np.array_equal(handoff, handoffs[0])
                        handoffs.append(handoff.copy())
                        k = n
                        while not (s.term or s.trunc):
                            if name == 'historical_actions':
                                action = source['actions'][k]; count = int(source['native_counts'][k])
                            else:
                                x = K.T.L.packet(s.env, s.obs, s.steps, 40000, s.previous, s.pn)
                                with torch.no_grad(): action = models[name].step(torch.tensor(x[None]))[0][0].numpy()
                                count = 100
                            s.step(action, count); k += 1
                        a = s.arrays(); a.update(initial_native_integration=s.initial.copy(),
                            handoff_native_integration=handoff, final_native_integration=K.R.integration(s.env).copy())
                        if name == 'historical_actions':
                            prefix_exact(a, source, len(source['states']))
                            assert np.array_equal(a['final_native_integration'], source['final_native_integration'])
                        row = dict(id=f'source{si}_{name}', source_id=item['row']['id'], suffix=name,
                                   prefix_decisions=n, prefix_physics=item['prefix_physics'], physics_steps=s.steps,
                                   complete=bool(s.term or s.trunc), success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),
                                   collision=bool(s.info.get('collision')), timeout=bool(s.trunc),
                                   remaining_m=float(s.env._goal_state()[1]), goal_xy=s.goal.tolist(), scene_sha256=s.scene,
                                   suffix_policy_hash=p['models'][name]['policy_hash'] if name in models else None,
                                   historical_failure_exact=name == 'historical_actions', not_autonomous_from_start=True,
                                   integration_sha256=K.R.arrays_hash(a['final_native_integration']),
                                   handoff_sha256=digest(handoff), array_hashes={key: digest(v) for key, v in a.items()})
                        if replica == 'primary':
                            b = io.BytesIO(); np.savez_compressed(b, **T.packed(a)); blob = b.getvalue()
                            assert len(blob) <= p['episode_bytes']
                            path = OUT / 'episodes' / (row['id'] + '.npz'); assert not path.exists(); path.write_bytes(blob)
                            row.update(path=str(path), sha256=K.C.sha(path)); rows.append(row); reference = dict(row=copy.deepcopy(row), arrays=a)
                            with (OUT / 'cases.jsonl').open('a') as f: f.write(json.dumps(row, separators=(',', ':')) + '\n')
                        else:
                            assert row['array_hashes'] == reference['row']['array_hashes']
                            assert all(np.array_equal(v, reference['arrays'][key]) for key, v in a.items())
                            row.update(full_arrays_and_integration_exact=True, independent_sample=False); reps.append(row)
                            with (OUT / 'replays.jsonl').open('a') as f: f.write(json.dumps(row, separators=(',', ':')) + '\n')
                        s.close(); s = None; status(True)
            state = 'COMPLETE'
    except BaseException:
        save(OUT / 'failure.json', dict(error=traceback.format_exc(), phase=phase, commands=commands, physics_steps=physics))
        if s is not None:
            try:
                a = s.arrays(); a['final_native_integration'] = K.R.integration(s.env).copy()
                b = io.BytesIO(); np.savez_compressed(b, **T.packed(a)); blob = b.getvalue()
                assert len(blob) <= p['partial_bytes']; (OUT / 'partial.npz').write_bytes(blob)
                save(OUT / 'partial.json', dict(phase=phase, physics_steps=s.steps, full_arrays={key: digest(v) for key, v in a.items()},
                                               missing_or_administratively_censored=True, not_navigation_failure=True))
            except BaseException: save(OUT / 'partial_failure.json', dict(error=traceback.format_exc()))
    finally:
        if s is not None: s.close()
        if w is not None: w.close()
        signal.alarm(0)
    result = dict(state=state, pid=os.getpid(), main_epoch=start, elapsed_s=time.time() - start,
                  deadline_epoch=deadline, commands=commands, physics_steps=physics, cases=rows, replays=reps,
                  verified_recovery_ids=[r['id'] for r in rows if r['success'] and not r['collision'] and r['suffix'] != 'historical_actions'
                                         and any(z['id'] == r['id'] and z['full_arrays_and_integration_exact'] for z in reps)],
                  Actor_updates=0, optimizer_calls=0, new_onpolicy_training=0, workers_spawned=0, all_workers_stopped=True,
                  original275=None, original_six54=None, fresh16=None, reload78=None, stable_six=False, entered7or8=False,
                  own_bytes=K.S.used(OUT), global_bytes=K.S.used(ROOT / 'stage/runs'))
    save(OUT / 'summary.json', result); save(OUT / 'status.json', result)
    print(json.dumps({key: v for key, v in result.items() if key not in ['cases', 'replays']}))


def audit():
    p = read(HERE / 'protocol.json'); verify(p); s = read(OUT / 'summary.json')
    assert not (DOC / 'T345_AUDIT.json').exists()
    for pid in [s['pid'], read(OUT / 'launch.json')['pid']]: assert not Path('/proc', str(pid)).exists()
    assert s['state'] == 'COMPLETE' and len(s['cases']) == len(s['replays']) == 6
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.optim.Adam.step = forbid
    models = {name: actor(p, name) for name in p['models']}; physics = 0; handoffs = {}
    for row in s['cases']:
        assert K.C.sha(row['path']) == row['sha256']; a = arrays(row['path'])
        assert {key: digest(v) for key, v in a.items()} == row['array_hashes']
        rep = next(r for r in s['replays'] if r['id'] == row['id'])
        assert rep['array_hashes'] == row['array_hashes'] and rep['full_arrays_and_integration_exact']
        assert rep['physics_steps'] == row['physics_steps']; physics += 2 * row['physics_steps']
        source = next(x for x in p['sources'] if x['row']['id'] == row['source_id'])
        ref = arrays(source['row']['path']); n = source['prefix_decisions']; prefix_exact(a, ref, n)
        assert int(a['native_counts'].astype(np.int64).sum()) == row['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        assert list(map(bool, a['end_flags'][-1])) == [row['collision'], row['success'], row['timeout']]
        assert np.array_equal(a['states'][1:], a['next_states'][:-1])
        assert np.array_equal(a['states'][n], ref['states'][n])
        assert row['handoff_sha256'] == digest(a['handoff_native_integration'])
        if row['source_id'] in handoffs: assert np.array_equal(handoffs[row['source_id']], a['handoff_native_integration'])
        handoffs[row['source_id']] = a['handoff_native_integration']
        if row['suffix'] == 'historical_actions':
            prefix_exact(a, ref, len(ref['states'])); assert np.array_equal(a['final_native_integration'], ref['final_native_integration'])
        else:
            q = models[row['suffix']]
            with torch.no_grad(): assert np.array_equal(q.step(torch.tensor(a['states'][n:]))[0].numpy(), a['actions'][n:])
        steps = np.r_[0, np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.abs(a['states'][:, 112] - (1 - steps / 40000)).max() < 1e-7
        for k, x in enumerate(a['states']):
            assert np.abs(K.H.rebuild(x, a['poses'][k], a['history_poses'][k], row['goal_xy'], a['history_goals'][k]) - x).max() <= 2e-6
            d0 = K.H.fields(a['poses'][k], row['goal_xy'])[1]; d1 = K.H.fields(a['next_poses'][k], row['goal_xy'])[1]
            nstep = int(a['native_counts'][k]); c, g, t = map(bool, a['end_flags'][k]); z = K.R.GAMMA ** nstep
            reward = -.000002 * (1 - z) / (1 - K.R.GAMMA) + .03 * d0 - z * .03 * d1 * (not (c or g or t))
            if c or g or t: reward += K.R.GAMMA ** (nstep - 1) * (float(g) - float(c) - .5 * float(t))
            assert abs(reward - a['rewards'][k]) < 1e-12
        assert K.R.arrays_hash(a['final_native_integration']) == row['integration_sha256']
    assert physics == s['physics_steps'] <= p['max_physics'] and s['commands'] == 13
    assert s['elapsed_s'] <= p['wall_seconds'] and not s['stable_six']
    save(DOC / 'T345_AUDIT.json', dict(state='PASS', native_added=0, optimizer_added=0,
         physics_steps=physics, commands=13, primary=6, exact_replays=6,
         original_failures_exact=True, same_real_native_handoffs=True, all120_history_clock_rewards_and_suffix_actions_verified=True,
         raw_and_model_SHA_exact=True, all_native_processes_stopped=True,
         verified_recovery_ids=s['verified_recovery_ids'], not_autonomous_six_acceptance=True,
         summary_sha256=K.C.sha(OUT / 'summary.json')))
    print(json.dumps(dict(state='AUDIT_PASS', native=0, Adam=0, verified_recovery_ids=s['verified_recovery_ids'])))


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'launch': launch, 'run': main, 'audit': audit}[sys.argv[1]]()
