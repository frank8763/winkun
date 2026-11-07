"""Independent, zero-native reconstruction of the registered joint pilot.

This audit never rolls out a simulator.  Optimizer steps below are discarded
reconstructions of recorded live steps, not additional navigation training.
"""
from t408_core import *
import t404_core as T408_NATIVE_AUDIT


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line] if Path(path).exists() else []


def model_readback(q, opt, record):
    assert sha(record['path']) == record['sha256']
    stored = torch.load(io.BytesIO(gzip.decompress(Path(record['path']).read_bytes())),
                        weights_only=False, map_location='cpu')
    actual = L.statehash(q, opt)
    assert {k: digest(v) for k, v in stored['state'].items()} == actual
    assert stored['hashes'] == record['hashes'] == actual
    assert stored['policy_hash'] == record['policy_hash'] == q.phash()


def joint_check(rows, role, q, p, record):
    selections = []
    for suffix, jobs, key in [('old8', p['old8_jobs'], 'old8'),
                              ('left5', p['left5_jobs'], 'left5'),
                              ('right', [p['right_job']], 'right1')]:
        selected = [r for r in rows if r['role'].startswith(role + '_' + suffix + '_')]
        assert [r['job_id'] for r in selected] == [j['id'] for j in jobs]
        assert len({r['job_id'] for r in selected}) == len(jobs)
        assert all(r['policy_hash'] == q.phash() for r in selected)
        assert tally(selected) == record[key]
        selections.extend(selected)
    safe_ids = sorted(r['job_id'] for r in selections if r['success'])
    accepted = all(r['success'] for r in selections)
    assert len(selections) == 14
    assert record['policy_hash'] == q.phash()
    assert record['success_ids'] == safe_ids and record['accepted'] == accepted
    return accepted


def raw_ledger(p, summary):
    rows = jsonl(OUT / 'cases.jsonl')
    replays = jsonl(OUT / 'replays.jsonl')
    commands = jsonl(OUT / 'commands.jsonl')
    partials = []
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = N.S.arrays(r['path'])
        assert int(a['native_counts'].sum()) == r['physics_steps'] == r['physics']
        assert bool(a['end_flags'][-1, 1]) == r['success']
        assert bool(a['end_flags'][:, 0].any()) == r['collision']
        assert bool(a['end_flags'][-1, 2]) == r['timeout']
        assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
        assert not (r['success'] and r['collision'])
        if r['full']:
            assert digest(a) == r['array_digest']
            assert digest(a['final_native_integration']) == r['integration_sha256']
            assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    for f in sorted(OUT.rglob('partial.npz')):
        a = N.S.arrays(f)
        partials.append(dict(path=str(f), sha256=sha(f),
                             recorded_physics=int(a['native_counts'].sum()),
                             administrative_censor=True, independent_sample=False))
    for r in replays:
        role = r['role']
        if role.startswith('train_') and '_exact_' in role:
            match = [x for x in rows if x['role'] == role.replace('_exact_', '_')]
        elif role.startswith('final_centres_exact_'):
            match = [x for x in rows if x['role'] == role.replace('final_centres_exact_', 'final_centres_', 1)]
        elif role.startswith('reload78_'):
            match = [x for x in rows if x['job_id'] == r['job_id'] and
                     any(x['role'].startswith(prefix) for prefix in ['main54_', 'old275_', 'fresh16_'])]
        else:
            raise AssertionError(('Undeclared replay role', role))
        assert len(match) == 1
        ref = match[0]
        assert r['exact'] and r['independent_sample'] is False
        for k in ['job_id', 'array_digest', 'integration_sha256', 'physics_steps',
                  'success', 'collision', 'timeout', 'policy_hash', 'seed']:
            assert r[k] == ref[k], (role, k)
    recorded = sum(r['physics_steps'] for r in rows + replays)
    ledger = dict(raw_cases=len(rows), exact_native_replays=len(replays),
                  command_records=len(commands), summary_commands=summary['commands'],
                  recorded_complete_physics=recorded,
                  recorded_partial_physics=sum(x['recorded_physics'] for x in partials),
                  summary_physics=summary['physics'], partials=partials,
                  complete_unledgered_physics=summary['physics'] - recorded,
                  unledgered_commands=summary['commands'] - len(commands))
    if summary['state'] == 'COMPLETE':
        assert not partials
        assert len(commands) == summary['commands'] == 4 + len(rows) + len(replays)
        assert [r['id'] for r in commands] == list(range(1, len(commands) + 1))
        assert sum(r['physics'] for r in commands) == summary['physics'] == recorded
        assert not any(r['error'] for r in commands)
        assert summary['commands'] + read(OUT / 'preflight.json')['zero_physics_commands'] <= p['max_commands']
        assert summary['physics'] <= p['max_physics']
        assert summary['elapsed_s'] <= p['main_seconds']
    return rows, replays, commands, ledger


def main():
    N.offline()
    p = plan()
    base = verify(p)
    s = read(OUT / 'summary.json')
    assert not (OUT / 'audit.json').exists(), 'Do not overwrite an existing audit'
    assert not Path('/proc', str(s['pid'])).exists()
    for pid in read(OUT / 'workers.json')['workers']:
        assert not Path('/proc', str(pid)).exists()
    # Check the pre-audit manifest before adding any audit artifact.
    for row in read(OUT / 'local_manifest.json')['files']:
        assert sha(ROOT / row['path']) == row['sha256']
    rows, replays, commands, ledger = raw_ledger(p, s)
    if s['state'] != 'COMPLETE':
        save(OUT / 'audit.json', dict(state='FAILED_PRODUCER_PARTIAL_PRESERVED',
             native=0, new_navigation_training=0, historical_discarded_Adam=0,
             raw_ledger=ledger, optimizer_reconstruction='NOT_EXECUTED',
             scientific_acceptance='NOT_ESTABLISHED'))
        N.manifest(OUT)
        raise AssertionError('Producer failed; raw partial ledger preserved, no PASS claim')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    q, opt = learner(base, p)
    broad = N.old_data(q, base, lambda: None)
    initial_old = [r for r in rows if r['role'].startswith('initial_old8_')]
    assert [r['job_id'] for r in initial_old] == [j['id'] for j in p['old8_jobs']]
    initial_arrays = []
    for r in initial_old:
        a = N.S.arrays(r['path'])
        assert r['full'] and r['success']
        assert r['policy_hash'] == q.phash()
        assert r['array_digest'] == p['expected_old8_array_digests'][r['job_id']]
        T408_NATIVE_AUDIT.audit_array(q, a, p, 0, None)
        assert not a['ownership'].any() and not a['stochastic'].any()
        initial_arrays.append(a)
    initial_right = [r for r in rows if r['role'].startswith('initial_right_')]
    initial_left = [r for r in rows if r['role'].startswith('initial_left5_')]
    assert len(initial_right) == 1 and initial_right[0]['success']
    assert [r['job_id'] for r in initial_left] == [j['id'] for j in p['left5_jobs']]
    assert all(r['policy_hash'] == q.phash() for r in initial_right + initial_left)
    initial = s['result']['initial']
    assert initial['old8'] == tally(initial_old)
    assert initial['right1'] == tally(initial_right)
    assert initial['left5'] == tally(initial_left)
    assert initial['old275'] is None and initial['main54'] is None
    g = groups(q, initial_arrays, p, broad)
    calls = 0
    before = L.state(q, opt)
    assert L.statehash(q, opt) == read(OUT / 'blocks/joint/before_hashes.json')
    actual = consolidate(q, opt, g, p, lambda: None, lambda *_: None)
    calls += p['consolidation_steps']
    warm = s['result']['joint_consolidation']
    assert warm == read(OUT / 'blocks/joint/result.json')
    assert actual == warm['fit']
    model_readback(q, opt, warm['proposed_model'])
    accepted = joint_check(rows, 'joint_proposal', q, p, warm['checks'])
    assert accepted == warm['accepted']
    if not accepted:
        L.load(q, opt, before)
    assert L.statehash(q, opt) == warm['retained_hashes']
    expected_events = []
    for which in ['live', 'shadow']:
        for k in range(p['consolidation_steps']):
            expected_events.extend([['joint', which, 'BEFORE', k], ['joint', which, 'AFTER', k]])
    if accepted:
        # The producer intentionally discards supervised momentum once, before RL.
        opt = torch.optim.Adam(N.trainable(q), lr=p['lr'])
        assert len(s['rounds']) == len(p['rl_jobs']) == 2
        for rnd, recorded in enumerate(s['rounds']):
            assert recorded['round'] == rnd and recorded['job'] == p['rl_jobs'][rnd]
            assert recorded == read(OUT / 'blocks' / str(rnd) / 'result.json')
            before = L.state(q, opt)
            assert L.statehash(q, opt) == read(OUT / 'blocks' / str(rnd) / 'before_hashes.json')
            batch = []
            train_rows = []
            for k in range(4):
                r = next(x for x in rows if x['role'] == f'train_{rnd}_{k}')
                assert r['full'] and r['job_id'] == p['rl_jobs'][rnd]['id']
                assert r['seed'] == [p['seed'], rnd, k] and r['policy_hash'] == q.phash()
                a = N.S.arrays(r['path'])
                T408_NATIVE_AUDIT.audit_array(q, a, p, 0, r['seed'])
                assert not a['ownership'].any()
                batch.append(a)
                train_rows.append(r)
            actual = rl_fit(q, opt, batch, g, p, rnd, lambda: None, lambda *_: None)
            calls += p['epochs']
            assert actual == recorded['proposal'] and tally(train_rows) == recorded['sampling']
            model_readback(q, opt, recorded['proposed_model'])
            safe = joint_check(rows, f'RL_{rnd}_proposal', q, p, recorded['checks'])
            keep = actual['numeric_accept'] and safe
            assert keep == recorded['accepted']
            if not keep:
                L.load(q, opt, before)
            assert L.statehash(q, opt) == recorded['retained_hashes']
            for which in ['live', 'shadow']:
                for k in range(p['epochs']):
                    expected_events.extend([[rnd, which, 'BEFORE', k], [rnd, which, 'AFTER', k]])
    else:
        assert s['rounds'] == []
        assert not any(r['role'].startswith('train_') for r in rows)
    assert jsonl(OUT / 'optimizer_calls.jsonl') == expected_events
    assert s['updates'] == dict(live=calls, shadow=calls)
    model_readback(q, opt, s['result']['model'])
    centres = [r for r in rows if r['role'].startswith('final_centres_')]
    assert [r['job_id'] for r in centres] == [j['id'] for j in p['centres']]
    assert all(r['policy_hash'] == q.phash() for r in centres)
    assert tally(centres) == s['result']['main_centres']
    if accepted:
        assert joint_check(rows, 'final_joint', q, p, s['result']['final_joint'])
        main_rows = [r for r in rows if r['role'].startswith('main54_')]
        old_rows = [r for r in rows if r['role'].startswith('old275_')]
        assert [r['job_id'] for r in main_rows] == [j['id'] for j in p['main_jobs']]
        assert [r['job_id'] for r in old_rows] == [j['id'] for j in base['final_old_jobs']]
        assert tally(main_rows) == s['result']['main54']
        assert tally(old_rows) == s['result']['old275']
        assert all(r['policy_hash'] == q.phash() for r in main_rows + old_rows)
    else:
        for key in ['main54', 'old275', 'fresh16', 'reload78']:
            assert s['result'][key] is None
        assert not s['result']['stable_wide_six']
    fresh = [r for r in rows if r['role'].startswith('fresh16_')]
    if fresh:
        assert accepted and all(r['success'] for r in main_rows + old_rows)
        assert [r['job_id'] for r in fresh] == [j['id'] for j in p['fresh_jobs']]
        assert tally(fresh) == s['result']['fresh16']
        assert all(r['policy_hash'] == q.phash() for r in fresh)
    else:
        assert s['result']['fresh16'] is None
    reloads = [r for r in replays if r['role'].startswith('reload78_')]
    stable = bool(fresh and all(r['success'] for r in fresh) and len(reloads) == 78)
    assert s['result']['stable_wide_six'] == stable
    if stable:
        assert [r['job_id'] for r in reloads] == [j['id'] for j in p['reload_jobs']]
        for key, count in [('main54', 54), ('old275', 275), ('fresh16', 16)]:
            assert s['result'][key]['success'] == s['result'][key]['valid'] == count
        assert s['result']['reload78'] == dict(exact=78, independent_sample=False)
    else:
        assert not reloads and s['result']['reload78'] is None
    expected_roles = [f'initial_old8_{j["id"]}' for j in p['old8_jobs']]
    expected_roles += [f'initial_right_{p["right_job"]["id"]}']
    expected_roles += [f'initial_left5_{j["id"]}' for j in p['left5_jobs']]
    def joint_roles(prefix):
        return ([f'{prefix}_old8_{j["id"]}' for j in p['old8_jobs']] +
                [f'{prefix}_left5_{j["id"]}' for j in p['left5_jobs']] +
                [f'{prefix}_right_{p["right_job"]["id"]}'])
    expected_roles += joint_roles('joint_proposal')
    if accepted:
        for rnd in range(len(p['rl_jobs'])):
            expected_roles += [f'train_{rnd}_{k}' for k in range(4)]
            expected_roles += joint_roles(f'RL_{rnd}_proposal')
    expected_roles += [f'final_centres_{j["id"]}' for j in p['centres']]
    if accepted:
        expected_roles += joint_roles('final_joint')
        expected_roles += [f'main54_{j["id"]}' for j in p['main_jobs']]
        expected_roles += [f'old275_{j["id"]}' for j in base['final_old_jobs']]
    if fresh:
        expected_roles += [f'fresh16_{j["id"]}' for j in p['fresh_jobs']]
    assert [r['role'] for r in rows] == expected_roles
    save(OUT / 'audit.json', dict(state='PASS', native=0, new_navigation_training=0,
         historical_discarded_Adam=calls, live_shadow_event_order_exact=True,
         actual_old8_hidden_history_rewards_and_initial_parent_exact=True,
         whole_Actor_Adam_proposals_rejections_and_fresh_Adam_exact=True,
         LEFT_RIGHT_joint_success_ID_gate_exact=True,
         full_training_raw_density_rewards_and_replays_exact=True,
         final_checkpoint_original_bytes_and_stored_Actor_Adam_exact=True,
         raw_ledger=ledger, physics=s['physics'], commands=s['commands'],
         old275=s['result']['old275'], main54=s['result']['main54'],
         fresh16=s['result']['fresh16'], reload78=s['result']['reload78'],
         stable_wide_six=stable, seven_eight_entered=False))
    N.manifest(OUT)
    print('T408_INDEPENDENT_AUDIT_PASS')


if __name__ == '__main__':
    main()
