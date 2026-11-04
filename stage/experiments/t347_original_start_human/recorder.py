"""User-requested earlier control: human from the original perturbed start."""
from pathlib import Path
import sys, importlib.util, copy, json, os, time, socket, contextlib, hashlib
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t347_original_start_human'; DOC = ROOT / 'stage/docs/review_20261104'
spec = importlib.util.spec_from_file_location('stage347_verified_human_engine', ROOT / 'stage/experiments/t346_actual_state_human/recorder.py')
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
K, R, W, M, np, torch = V.K, V.R, V.W, V.M, V.np, V.torch
read, save = V.read, V.save
V.HERE = M.HERE = W.HERE = R.HERE = HERE
V.OUT = M.OUT = W.OUT = R.OUT = OUT
Handler = V.Handler


class App(W.App):
    # W.App.begin has no policy prefix: only a real user request advances time.
    def finish_attempt(self, reason):
        super().finish_attempt(reason)
        for row in self.attempts:
            row.update(source='REAL_HUMAN_FROM_ORIGINAL_PERTURBED_START', human_start_decision=0,
                       policy_prefix_decisions=0, not_autonomous_navigation=True)
        save(OUT / 'attempts.json', self.attempts)


M.App = App


def prepare():
    assert not OUT.exists() and read(DOC / 'T346_AUDIT.json')['state'] == 'PASS'
    receipt = ROOT / 'stage/docs/long_distance_20261101/release_receipt_T346.json'
    assert read(receipt)['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p = copy.deepcopy(read(ROOT / 'stage/experiments/t346_actual_state_human/protocol.json'))
    p.update(experiment='T347_ORIGINAL_START_HUMAN_AFTER_LATE_HANDOFF_FEEDBACK', port=18772,
             prefix_decisions=0, prefix_seconds=0.,
             course_id='LEFT6_WIDE80_DX_MINUS003_REAL_HUMAN_FROM_START',
             contract='New user-steered correction of late handoff design. One real human full episode from wide_dev1 originalstart(dx-.03),0policy/nativeprefix,full80s unchanged wheel5force120/gap1.17m/no reverse. All actions originate in current human UI requests; no scripted teacher, invented expert or logp. One attempt+full12native replay+1zero-physics geometry=3commands80kallphysics1200s last240/own3MiB/global1GiB/free40GiB. No automatic retry or oldT346 restart; raw failure preserved. Human labels are all actual actions from index0; no Actor/Q/Adam/27554fresh16reload78/7or8/planner/map/waypoints/Phase. This changes who controls the beginning,not vehicle dynamics. Original narrow six stability not measured. User currently available and actively reports previous handoff too late/no reverse; no repeated availability question.')
    p.pop('handoff_reference', None)
    files = set(p['input_hashes']) | {str(f) for f in HERE.iterdir() if f.suffix in ['.py', '.html', '.js']}
    files |= {str(DOC / 'T347_PLAN.md'), str(DOC / 'T346_AUDIT.json'), str(receipt)}
    p['input_hashes'] = {f: K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT / 'stage/runs') + p['final_bytes'] <= p['global_bytes']
    OUT.mkdir(); save(HERE / 'protocol.json', p); save(OUT / 'protocol.json', p)
    print(json.dumps(dict(state='REGISTERED_NOT_LAUNCHED', policy_prefix=0, human_sim_seconds=80,
                         reserved_global=K.S.used(ROOT / 'stage/runs') - K.S.used(OUT) + p['final_bytes'])))


def preflight():
    p = read(HERE / 'protocol.json'); M.install(p); R.verify(p)
    assert not (OUT / 'preflight.json').exists(); torch.set_num_threads(1); torch.optim.Adam.step = R.U.forbid_step
    with socket.socket() as check: check.bind(('127.0.0.1', p['port']))
    folder = OUT / 'preflight'; folder.mkdir(); s = view = None
    with (folder / 'factory.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        native, create, w, reward = M.setup(folder / 'native')
        try:
            s = R.Session(p, native, create, w, reward, lambda: None,
                          lambda: (_ for _ in ()).throw(AssertionError('Zero preflight native')))
            a, initial, _ = V.source_arrays(p)
            assert np.array_equal(s.initial, initial) and np.array_equal(s.initial120, a['states'][0])
            assert s.steps == 0 and s.env.max_steps == 40000 and s.env.max_wheel == 5.
            commands = [s.env._policy_to_commands(np.array([v, 0.], np.float32))[1] for v in [-1., 0., 1.]]
            assert min(commands) >= 0. and commands[-1] == 5.
            view = R.View(s); assert len(view.frame(s)) > 1000 and s.steps == 0
            boundary = "  if(typeof document==='undefined')return;"
            old = ROOT / 'stage/experiments/t331_policy_prefix_retry/control.js'
            cls = (HERE / 'control.js').read_text().split(boundary)[0]
            assert cls == old.read_text().split(boundary)[0]
            prior = ROOT / 'stage/runs/t331_policy_prefix_retry/client_contract.json'
            assert read(prior)['state'].startswith('PASS')
            save(OUT / 'client_contract.json', dict(state='PASS_REUSED_IDENTICAL_CLIENT_CONTROL_CLASS',
                 source_contract=str(prior), source_contract_sha256=K.C.sha(prior),
                 identical_control_class_sha256=hashlib.sha256(cls.encode()).hexdigest(),
                 new_client_test_executed=False, native_commands=0, physics_steps=0, human_actions=0))
            save(OUT / 'preflight.json', dict(state='PASS', geometry_commands=1, physics_steps=0, Adam_calls=0,
                 initial120_and_native_exact=True, copied_render_native_unchanged=True, policy_prefix_decisions=0,
                 raw_drive_minus1_zero_plus1_wheel_commands=commands, reverse_available=False,
                 source_protocol_sha256=K.C.sha(HERE / 'protocol.json')))
        finally:
            if view is not None: view.close()
            if s is not None: s.close()
            w.close()
    print('T347 ready0physics0Adam,originalstart human80s,no policy prefix')


def audit():
    p = read(HERE / 'protocol.json'); R.verify(p); s = read(OUT / 'summary.json'); dest = DOC / 'T347_AUDIT.json'
    assert not dest.exists() and s['state'] == 'COMPLETE'
    for pid in [s['pid'], read(OUT / 'launch.json')['pid']]: assert not Path('/proc', str(pid)).exists()
    torch.set_num_threads(1); torch.optim.Adam.step = R.U.forbid_step
    ref, initial, _ = V.source_arrays(p); physics = 0
    events = [json.loads(x) for x in (OUT / 'request_events.jsonl').read_text().splitlines()] if (OUT / 'request_events.jsonl').exists() else []
    inputs = [json.loads(x) for x in (OUT / 'human_inputs.jsonl').read_text().splitlines()] if (OUT / 'human_inputs.jsonl').exists() else []
    assert len({e['request_id'] for e in events}) == len(events)
    for row in s['attempts']:
        assert K.C.sha(row['path']) == row['sha256']; a, raw_initial, final = R.load_arrays(row['path'])
        assert np.array_equal(raw_initial, initial) and {k: K.S.digest(v) for k, v in a.items()} == row['array_hashes']
        assert row['human_start_decision'] == 0 and row['policy_prefix_decisions'] == 0
        assert np.array_equal(a['actions'], np.array([e['action'] for e in events], np.float32).reshape(-1, 2))
        assert np.array_equal(a['actions'], np.array([e['action'] for e in inputs], np.float32).reshape(-1, 2))
        assert [e['decision'] for e in events] == list(range(len(a['actions'])))
        assert int(a['native_counts'].astype(np.int64).sum()) == row['physics_steps']; physics += row['physics_steps']
        assert K.R.arrays_hash(final) == row['integration_sha256']
        if len(a['states']):
            assert np.array_equal(a['states'][0], ref['states'][0]) and np.array_equal(a['states'][1:], a['next_states'][:-1])
            assert not a['end_flags'][:-1].any()
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
    save(dest, dict(state='PASS', native_added=0, optimizer_added=0, all_actions_are_real_human_from_start=True,
         unique_requests_full12_history_clock_rewards_and_native_replay_exact=True,
         commands=s['commands'], physics_steps=physics, not_autonomous_six_acceptance=True,
         summary_sha256=K.C.sha(OUT / 'summary.json')))


if __name__ == '__main__':
    {'prepare': prepare, 'preflight': preflight, 'launch': V.launch, 'main': M.main, 'audit': audit}[sys.argv[1]]()
