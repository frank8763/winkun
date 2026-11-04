"""One real human correction from the fixed T345 consumed failure state."""
from pathlib import Path
import sys, importlib.util, copy, json, os, subprocess, time, socket, contextlib, hashlib
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t346_actual_state_human'; DOC = ROOT / 'stage/docs/review_20261104'
spec = importlib.util.spec_from_file_location('stage346_existing_human_engine', ROOT / 'stage/experiments/t331_policy_prefix_retry/recorder.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
K, R, W, np, torch = M.K, M.R, M.W, M.np, M.torch
read, save = M.read, K.C.save
M.HERE = W.HERE = R.HERE = HERE
M.OUT = W.OUT = R.OUT = OUT
M.DOC = DOC
Handler = M.Handler


def source_arrays(p):
    a, initial, final = R.load_arrays(p['reference'])
    return a, initial, final


class App(W.App):
    def begin(self):
        super().begin()
        self.state = 'POLICY_PREFIX'; self.write_status()
        a, initial, _ = source_arrays(self.p); s = self.session
        assert np.array_equal(s.initial, initial)
        for k in range(self.p['prefix_decisions']):
            s.step(a['actions'][k], int(a['native_counts'][k]))
            assert not (s.term or s.trunc)
        actual = s.arrays(); n = self.p['prefix_decisions']
        for name, value in actual.items():
            stop = n + 1 if name.startswith('history_') else n
            assert np.array_equal(value, a[name][:stop]), name
        with np.load(self.p['handoff_reference']) as z:
            assert np.array_equal(K.R.integration(s.env), z['handoff_native_integration'])
        assert s.steps == 7600 and len(s.rows) == 76
        self.checkpoint(); self.jpg = self.view.frame(s); self.state = 'HUMAN_RECORDING'
        save(OUT / 'handoff.json', dict(prefix_decisions=76, physics_steps=s.steps,
             integration_sha256=K.R.arrays_hash(K.R.integration(s.env)),
             state_sha256=K.S.digest(K.T.L.packet(s.env, s.obs, s.steps, 40000, s.previous, s.pn)),
             prefix_is_recorded_learner_actions_not_human=True, no_reset_at_handoff=True))
        self.write_status()

    def finish_attempt(self, reason):
        super().finish_attempt(reason)
        for row in self.attempts:
            row.update(source='REAL_LEARNER_FAILURE_PREFIX_THEN_EXPLICIT_HUMAN_CORRECTION',
                       human_start_decision=self.p['prefix_decisions'],
                       prefix_source_id='T344_training_r0_n2', not_autonomous_navigation=True)
        save(OUT / 'attempts.json', self.attempts)


M.App = App


def prepare():
    assert not OUT.exists()
    assert read(DOC / 'T345_AUDIT.json')['state'] == 'PASS'
    receipt = ROOT / 'stage/docs/long_distance_20261101/release_receipt_T345.json'
    assert read(receipt)['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert read(ROOT / 'stage/runs/t345_verified_recovery/summary.json')['verified_recovery_ids'] == []
    p = copy.deepcopy(read(ROOT / 'stage/experiments/t331_policy_prefix_retry/protocol.json'))
    p.update(experiment='T346_FIXED_REAL_FAILURE_STATE_HUMAN_CORRECTION', port=18771,
             job=next(j for j in read(ROOT / 'stage/experiments/t344_neighbour_recovery/protocol.json')['course_dev_jobs'] if j['id'] == 'wide_dev_1'),
             initial_geometry=read(ROOT / 'stage/runs/t345_verified_recovery/geometry.json'),
             reference=str(ROOT / 'stage/runs/t344_neighbour_recovery/episodes/training_r0_n2.npz'),
             handoff_reference=str(ROOT / 'stage/runs/t345_verified_recovery/episodes/source1_historical_actions.npz'),
             prefix_decisions=76, prefix_seconds=15.2, max_attempts=1,
             max_commands=3, max_main_commands=2, max_physics=80000, wall_seconds=1200,
             final_reserve_seconds=240, audit_seconds=300, all_work_seconds=2400,
             preflight_geometry_commands=1, final_bytes=3 * 1024**2,
             payload_bytes=1536 * 1024, metadata_bytes=1536 * 1024, episode_bytes=768 * 1024,
             course_id='LEFT6_WIDE80_ACTUAL_LEARNER_ERROR_PREFIX15P2_REAL_HUMAN_RECOVERY',
             contract='One user-available human session only. Real consumed T344r0n2 first76 actions/7600native from wide_dev1 originalstart,all12/history/nativehandoff exact auditedT345. Pause at actual15.2seconds preserving clock/history/goal/actuators; afterthat exclusively real user steer/drive .999-domain,not synthetic expert. One attempt+full12native replay+1zero-physics geometry=3commands80kallphysics1200s last240reserve/own3MiB/global1GiB/free40GiB. Original wide1.17m80s/wheel5force120 unchanged. Prefix is not human labels; mixedtrajectory not onpolicy or autonomous6acceptance. No Actor/Q/Adam/newRL/27554fresh16reload78/7or8/planner/map/waypoints/teacherdeployment/router/Phase. No automatic retry,wait for current human availability before launch. Exact client control class,request dedup/Origin/sequence preserved; current snapshot and full evidence retained.')
    files = set(p['input_hashes']) | {str(f) for f in HERE.iterdir() if f.suffix in ['.py', '.html', '.js']}
    files |= {p['reference'], p['handoff_reference'], str(DOC / 'T346_PLAN.md'), str(DOC / 'T345_AUDIT.json'), str(receipt),
              str(ROOT / 'stage/runs/t345_verified_recovery/geometry.json'),
              str(ROOT / 'stage/runs/t331_policy_prefix_retry/client_contract.json')}
    p['input_hashes'] = {f: K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT / 'stage/runs') + p['final_bytes'] <= p['global_bytes']
    OUT.mkdir(); save(HERE / 'protocol.json', p); save(OUT / 'protocol.json', p)
    print(json.dumps(dict(state='REGISTERED_NOT_LAUNCHED', prefix=15.2, human_time_available=64.8,
                         reserved_global=K.S.used(ROOT / 'stage/runs') - K.S.used(OUT) + p['final_bytes'])))


def preflight():
    p = read(HERE / 'protocol.json'); M.install(p); R.verify(p)
    assert not (OUT / 'preflight.json').exists()
    torch.set_num_threads(1); torch.optim.Adam.step = R.U.forbid_step
    with socket.socket() as check: check.bind(('127.0.0.1', p['port']))
    folder = OUT / 'preflight'; folder.mkdir()
    s = view = None
    with (folder / 'factory.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        native, create, w, reward = M.setup(folder / 'native')
        try:
            s = R.Session(p, native, create, w, reward, lambda: None,
                          lambda: (_ for _ in ()).throw(AssertionError('Zero preflight native')))
            a, initial, _ = source_arrays(p)
            assert np.array_equal(s.initial, initial) and np.array_equal(s.initial120, a['states'][0])
            assert s.steps == 0 and s.env.max_steps == 40000 and s.env.max_wheel == 5.
            view = R.View(s); image = view.frame(s); assert len(image) > 1000 and s.steps == 0
            old = ROOT / 'stage/experiments/t331_policy_prefix_retry/control.js'
            boundary = "  if(typeof document==='undefined')return;"
            old_class = old.read_text().split(boundary)[0]
            new_class = (HERE / 'control.js').read_text().split(boundary)[0]
            assert old_class == new_class and 'class HumanControl' in new_class
            prior = ROOT / 'stage/runs/t331_policy_prefix_retry/client_contract.json'
            assert read(prior)['state'].startswith('PASS')
            save(OUT / 'client_contract.json', dict(state='PASS_REUSED_IDENTICAL_CLIENT_CONTROL_CLASS',
                 source_contract=str(prior), source_contract_sha256=K.C.sha(prior),
                 identical_control_class_sha256=hashlib.sha256(new_class.encode()).hexdigest(),
                 source_full_js_sha256=K.C.sha(old), current_full_js_sha256=K.C.sha(HERE / 'control.js'),
                 new_client_test_executed=False,
                 native_commands=0, physics_steps=0, human_actions=0))
            save(OUT / 'preflight.json', dict(state='PASS', geometry_commands=1, physics_steps=0, Adam_calls=0,
                 actual_source_initial120_and_native_exact=True, copied_render_native_unchanged=True,
                 handoff_contract_source='T345 audited exact same76decision nativeprefix,not new preflight replay',
                 user_availability_required_before_launch=True, source_protocol_sha256=K.C.sha(HERE / 'protocol.json')))
        finally:
            if view is not None: view.close()
            if s is not None: s.close()
            w.close()
    print('T346 ready0physics0Adam,copiedrender and identical client;notlaunched')


def launch():
    p = read(HERE / 'protocol.json'); R.verify(p)
    assert read(OUT / 'preflight.json')['state'] == 'PASS' and read(OUT / 'http_contract.json')['state'] == 'PASS_DUMMY_HTTP_ONLY'
    availability = read(OUT / 'human_availability.json')
    assert availability['confirmed_by_user'] and availability['source'] == 'CURRENT_CHAT_EXPLICIT_AVAILABILITY_REPLY'
    assert 0 <= time.time() - availability['confirmed_epoch'] < 600
    assert not any((OUT / n).exists() for n in ['launch.json', 'status.json', 'summary.json'])
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip()
    receipt = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    epoch = time.time()
    with (OUT / 'run.log').open('xb') as log:
        proc = subprocess.Popen(['systemd-inhibit', '--what=sleep', '--mode=block', '--why=Stage actual failure human correction',
            'timeout', '--signal=TERM', '--kill-after=1s', str(p['wall_seconds']) + 's', 'taskset', '-c', '0',
            str(ROOT / 'run_stage.sh'), str(HERE / 'recorder.py'), 'main'], cwd=ROOT,
            env=dict(os.environ, STAGE_PROCESS_CPU='0'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT / 'launch.json', dict(pid=proc.pid, launch_epoch=epoch, deadline_epoch=epoch + p['wall_seconds'],
         all_work_deadline_epoch=epoch + p['all_work_seconds'], source_remote_commit=receipt['remote_commit']))
    print(json.dumps(dict(state='LAUNCHED', pid=proc.pid, url='http://127.0.0.1:' + str(p['port']))))


def audit():
    p = read(HERE / 'protocol.json'); R.verify(p); s = read(OUT / 'summary.json')
    dest = DOC / 'T346_AUDIT.json'; assert not dest.exists() and s['state'] == 'COMPLETE'
    for pid in [s['pid'], read(OUT / 'launch.json')['pid']]: assert not Path('/proc', str(pid)).exists()
    torch.set_num_threads(1); torch.optim.Adam.step = R.U.forbid_step
    ref, initial, _ = source_arrays(p); physics = 0
    events = [json.loads(x) for x in (OUT / 'request_events.jsonl').read_text().splitlines()] if (OUT / 'request_events.jsonl').exists() else []
    inputs = [json.loads(x) for x in (OUT / 'human_inputs.jsonl').read_text().splitlines()] if (OUT / 'human_inputs.jsonl').exists() else []
    assert len({e['request_id'] for e in events}) == len(events)
    for row in s['attempts']:
        assert K.C.sha(row['path']) == row['sha256']; a, raw_initial, final = R.load_arrays(row['path']); n = p['prefix_decisions']
        assert np.array_equal(raw_initial, initial) and {k: K.S.digest(v) for k, v in a.items()} == row['array_hashes']
        for name in a:
            stop = n + 1 if name.startswith('history_') else n
            assert np.array_equal(a[name][:stop], ref[name][:stop]), name
        handoff_state = a['states'][n] if len(a['states']) > n else a['next_states'][n - 1]
        assert np.array_equal(handoff_state, ref['states'][n])
        assert np.array_equal(a['actions'][n:], np.array([e['action'] for e in events], np.float32).reshape(-1, 2))
        assert np.array_equal(a['actions'][n:], np.array([e['action'] for e in inputs], np.float32).reshape(-1, 2))
        assert [e['decision'] for e in events] == list(range(n, len(a['actions'])))
        assert [e['decision'] for e in inputs] == list(range(n, len(a['actions'])))
        assert int(a['native_counts'].astype(np.int64).sum()) == row['physics_steps']; physics += row['physics_steps']
        assert K.R.arrays_hash(final) == row['integration_sha256']
        assert np.array_equal(a['states'][1:], a['next_states'][:-1]) and not a['end_flags'][:-1].any()
        steps = np.r_[0, np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.abs(a['states'][:, 112] - (1 - steps / 40000)).max() < 1e-7
        for k, x in enumerate(a['states']):
            assert np.abs(K.H.rebuild(x, a['poses'][k], a['history_poses'][k], row['goal_xy'], a['history_goals'][k]) - x).max() <= 2e-6
            d0 = K.H.fields(a['poses'][k], row['goal_xy'])[1]; d1 = K.H.fields(a['next_poses'][k], row['goal_xy'])[1]
            count = int(a['native_counts'][k]); c, g, t = map(bool, a['end_flags'][k]); z = K.R.GAMMA ** count
            reward = -.000002 * (1 - z) / (1 - K.R.GAMMA) + .03 * d0 - z * .03 * d1 * (not (c or g or t))
            if c or g or t: reward += K.R.GAMMA ** (count - 1) * (float(g) - float(c) - .5 * float(t))
            assert abs(reward - a['rewards'][k]) < 1e-12
        rep = next(r for r in s['replays'] if r['id'] == row['id'])
        assert rep['all12_arrays_exact'] and rep['physics_steps'] == row['physics_steps']; physics += rep['physics_steps']
    assert physics == s['physics_steps'] <= p['max_physics']
    save(dest, dict(state='PASS', native_added=0, optimizer_added=0, prefix_not_human_labels=True,
         all_human_actions_exact_to_unique_requests=True, full12_arrays_hash_and_native_replay_exact=True,
         commands=s['commands'], physics_steps=physics, not_autonomous_six_acceptance=True,
         summary_sha256=K.C.sha(OUT / 'summary.json')))


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'launch': launch, 'main': M.main, 'audit': audit}[sys.argv[1]]()
