"""Independent-process zero-native contracts for all three Stage relay profiles.

Real complete source-policy observation histories are fixed fixtures. The iid
actions and rewards used for PPO below are ENGINEERING SYNTHETIC data: those
actions have never driven these observations through an environment. They do
not establish navigation, on-policy sampling, reward validity, or policy gain.
Each profile/process spends three discarded PPO Adam steps. Iterative recovery
spends a separate three discarded BC steps:24 across all six A/B processes.
Import is inert; only explicit probe/compare entry points read or write files.
"""
from pathlib import Path
import copy
import os
import platform
import sys
import time
import t439_learning as L

E, torch, np = L.E, L.torch, L.np
ROOT = Path(__file__).resolve().parents[3]
STAGES = ('representation_rl', 'iterative_recovery', 'representation_rl')


def _runtime():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=str(np.__version__), default_dtype=str(torch.get_default_dtype()),
                intraop_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision(),
                cpu_autocast=torch.is_autocast_enabled('cpu'),
                grad_enabled=torch.is_grad_enabled(),
                OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
                MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
                OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'))


def _frozen(p):
    q = E.safe_initial(E.N.science(p), p).eval()
    q.sample = False
    q.reset_noise()
    L.configure(q, p)
    return q


def _mapping(q, opt):
    variables = L.params(q)
    names = {id(v): n for n, v in q.named_parameters()}
    actual = [id(v) for g in opt.param_groups for v in g['params']]
    assert actual == [id(v) for v in variables]
    assert len(actual) == len(set(actual))
    assert sum(v.numel() for v in variables) == L.trainable_count(q)
    prefixes = L.trainable_prefixes(q)
    assert all(v.requires_grad == n.startswith(prefixes) for n, v in q.named_parameters())
    return [dict(index=i, name=names[id(v)], shape=list(v.shape), dtype=str(v.dtype))
            for i, v in enumerate(variables)]


def fixture_sources(p):
    rows = p['current_anchor_rows']
    assert len(rows) == len({r['job_id'] for r in rows}) == 59
    assert {r['policy_hash'] for r in rows} == {p['expected_policy_hash']}
    old = {j['id'] for j in p['old8_jobs']}
    six = {j['id'] for j in p['main_jobs']}
    selected = [next(r for r in rows if r['job_id'] in old),
                next(r for r in rows if r['job_id'] in six)]
    target_file, target_audit = p['verified_target_file'], p['verified_target_audit']
    assert E.sha(target_file) == p['inputs'][target_file]
    assert E.sha(target_audit) == p['inputs'][target_audit]
    assert E.read(target_audit)['state'] == 'PASS'
    targets = E.read(target_file)
    assert len(targets) == 7
    target = next(t for t in targets if t['request']['mode'] == 'feedback')
    assert target['exact'] and target['row']['success'] and not target['row']['collision']
    assert target['row']['policy_hash'] == p['expected_policy_hash']
    return selected, target


def _authenticated_fixture(q, row, p, first):
    assert row['full'] and row['success'] and not row['collision'] and not row['timeout']
    assert E.sha(row['path']) == row['sha256']
    arrays = E.N.S.arrays(row['path'])
    assert E.digest(arrays) == row['array_digest']
    for key, expected in row['array_hashes'].items():
        assert E.digest(arrays[key]) == expected, key
    assert E.digest(arrays['final_native_integration']) == row['integration_sha256']
    before = E.digest(q.state_dict())
    E.audit_array(q, arrays, p, first, None)
    assert E.digest(q.state_dict()) == before
    return arrays


def _synthetic_batch(q, original, seeds=(4340101, 4340102, 4340103, 4340104)):
    """Generate policy tape on fixed real states, WITHOUT environment dynamics."""
    before = E.digest(q.state_dict())
    items, receipts = [], []
    x = torch.tensor(original['states'])
    for index, seed in enumerate(seeds):
        a = copy.deepcopy(original)
        q.reset_noise()
        q.rng = np.random.default_rng(np.random.SeedSequence(seed))
        hidden = None
        hidden_rows, commands = [], []
        with torch.no_grad():
            for row in x:
                hidden_rows.append(np.zeros(16) if hidden is None else hidden[0, 0].numpy().copy())
                q.sample = True
                action, _, hidden, _ = q.step(row[None], hidden)
                commands.append(action[0].numpy().copy())
            step_locations = np.asarray([row['locations'] for row in q.tape])
            captured_hidden = []
            def capture_core(module, args, output):
                captured_hidden.append((output[0].detach().clone(), output[1].detach().clone()))
            hook = q.core.register_forward_hook(capture_core)
            try:
                sequence = q.sequence(x, q.base_actions(x))
            finally:
                hook.remove()
            assert len(captured_hidden) == 1
            sequence_hidden = captured_hidden[0][0][:, 0].numpy()
            sequence_before = np.concatenate([np.zeros((1, 16)), sequence_hidden[:-1]], axis=0)
            hidden_error = float(np.abs(sequence_before - np.asarray(hidden_rows)).max())
            final_hidden_error = float(np.abs(captured_hidden[0][1].numpy()-hidden.numpy()).max())
            assert hidden_error < 1e-10 and final_hidden_error < 1e-10
            mean_error = float((sequence - torch.tensor(step_locations)).abs().max())
            assert mean_error < 1e-10
        for key in q.tape[0]:
            a[key] = np.asarray([row[key] for row in q.tape])
        a['locations'] = step_locations
        a['actions'] = np.asarray(commands)
        a['hidden_before'] = np.asarray(hidden_rows)
        a['last_hidden_after'] = hidden[0, 0].detach().numpy().copy()
        a['stochastic'] = np.ones(len(x), np.uint8)
        a['ownership'] = np.zeros(len(x), np.uint8)
        # These deliberately varied synthetic terminal rewards ensure the PPO
        # score gradient is exercised. They are NOT a reward reconstruction.
        a['rewards'] = np.asarray(original['rewards']).copy()
        a['rewards'][-1] += index * .1
        assert np.array_equal(a['end_flags'], original['end_flags'])
        with torch.no_grad():
            density = E.L.logp(sequence, torch.tensor(a['raw_actions']),
                               torch.tensor(a['atom']), q.contract)
            density_error = float((density - torch.tensor(a['logp'])).abs().max())
            assert density_error < 1e-9
        receipts.append(dict(seed=seed, arrays_sha256=E.digest(a),
                             step_sequence_mean_error=mean_error,
                             step_sequence_hidden_before_error=hidden_error,
                             step_sequence_final_hidden_error=final_hidden_error,
                             sequence_iid_logdensity_error=density_error,
                             synthetic_terminal_reward_increment=index * .1,
                             physical_observation_rollout=False))
        items.append(a)
    q.sample = False
    q.reset_noise()
    assert E.digest(q.state_dict()) == before
    return items, receipts


def _transaction(q, opt, g, pp, kind, original, guard, permitted, event_factory):
    """One fresh step and an exact mature live/shadow step, then rollback tests."""
    mapping, fixed = _mapping(q, opt), L.fixedhash(q)
    initial = copy.deepcopy(E.L.state(q, opt))
    initial_hash = E.L.statehash(q, opt)
    assert not opt.state
    synthetic = []

    def fit(model, optimizer, which, items=None):
        event = event_factory(which, model, optimizer)
        if kind == 'PPO':
            assert items is not None
            return L.fit_ppo(model, optimizer, items, g, pp, 0, guard, event, steps=1)
        assert kind == 'BC'
        return L.fit_bc(model, optimizer, g, pp, guard, event, steps=1)

    fresh_items = None
    if kind == 'PPO':
        fresh_items, receipt = _synthetic_batch(q, original)
        synthetic.append(dict(role='fresh', streams=receipt))
    fresh_fit = fit(q, opt, kind + '_fresh', fresh_items)
    mature_before = copy.deepcopy(E.L.state(q, opt))
    mature_before_hash = E.L.statehash(q, opt)
    assert opt.state and all(float(v['step']) == 1 for v in opt.state.values())
    assert initial_hash != mature_before_hash
    shadow = _frozen(pp)
    so = L.make_optimizer(shadow, pp)
    permitted.append(so)
    E.L.load(shadow, so, mature_before)
    assert _mapping(shadow, so) == mapping
    assert E.L.statehash(shadow, so) == mature_before_hash
    mature_items = None
    if kind == 'PPO':
        mature_items, receipt = _synthetic_batch(q, original)
        synthetic.append(dict(role='mature', streams=receipt))
    live_fit = fit(q, opt, kind + '_mature_live', mature_items)
    mature_after = copy.deepcopy(E.L.state(q, opt))
    mature_after_hash = E.L.statehash(q, opt)
    shadow_fit = fit(shadow, so, kind + '_mature_shadow', mature_items)
    assert live_fit == shadow_fit
    assert mature_after_hash == E.L.statehash(shadow, so)
    assert mature_before_hash != mature_after_hash
    assert L.fixedhash(q) == L.fixedhash(shadow) == fixed
    E.L.load(q, opt, mature_before)
    rejection = E.L.statehash(q, opt)
    assert rejection == mature_before_hash

    class SimulatedInterruption(Exception):
        pass

    try:
        E.L.load(q, opt, mature_after)
        assert E.L.statehash(q, opt) == mature_after_hash
        raise SimulatedInterruption('discarded mature proposal rollback contract')
    except SimulatedInterruption:
        E.L.load(q, opt, mature_before)
    interruption = E.L.statehash(q, opt)
    assert interruption == mature_before_hash and L.fixedhash(q) == fixed
    E.L.load(q, opt, initial)
    assert E.L.statehash(q, opt) == initial_hash
    return dict(kind=kind, mapping=mapping, frozen_hash=fixed,
                initial_hashes=initial_hash, mature_before_hashes=mature_before_hash,
                mature_after_hashes=mature_after_hash,
                shadow_after_hashes=E.L.statehash(shadow, so),
                rejection_restored=rejection, interruption_restored=interruption,
                Actor_Adam_rejection_and_interruption_exact=True,
                fresh_fit=fresh_fit, mature_live_fit=live_fit,
                mature_shadow_fit=shadow_fit, synthetic_streams=synthetic,
                discarded_fresh_Adam=1, discarded_mature_live_Adam=1,
                discarded_mature_shadow_Adam=1, discarded_Adam_total=3)


def probe(pfile, stage, label, outfile):
    assert stage in STAGES and label in ['A', 'B']
    pfile, outfile = Path(pfile).resolve(), Path(outfile).resolve()
    assert not outfile.exists(), ('Do not repeat completed probe', outfile)
    p = E.read(pfile)
    assert E.sha(p['student_start_model']) == p['student_start_sha256']
    for filename, expected in p['inputs'].items():
        assert E.sha(filename) == expected, filename
    pp = dict(p, relay_stage=stage, preflight_clone_probe=True)
    started = time.time()
    deadline = min(started + 300, p['deadline_all'], p['deadline_learning'])
    E.N.offline()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def guard():
        assert time.time() < deadline - 5

    def forbidden(*args, **kwargs):
        raise AssertionError('T439 clone contract prohibits native and model export')

    E.L.Session = forbidden
    E.Pool = forbidden
    E.N.S.P.C.B.setup = forbidden
    E.export = forbidden
    E.N.write_npz = forbidden
    L.export_model = forbidden
    for optimizer in [torch.optim.AdamW, torch.optim.SGD]:
        optimizer.step = forbidden

    q = _frozen(pp)
    assert q.phash() == p['expected_policy_hash']
    before_audit = E.digest(q.state_dict())
    anchor_rows, target = fixture_sources(p)
    anchors, source_receipts = [], []
    for row in anchor_rows:
        guard()
        assert row['loaded_model_sha256'] == p.get('anchor_source_model_sha256', p['student_start_sha256'])
        arrays = _authenticated_fixture(q, row, p, 0)
        anchors.append((arrays, row))
        source_receipts.append(dict(kind='existing_full_source_policy_anchor',
                                    id=row['job_id'], path=row['path'],
                                    sha256=row['sha256'], array_digest=row['array_digest']))
    guard()
    target_arrays = _authenticated_fixture(q, target['row'], p, target['first'])
    targets = [dict(arrays=target_arrays, first=target['first'], id=target['id'])]
    source_receipts.append(dict(kind='existing_verified_feedback_suffix',
                                id=target['id'], path=target['row']['path'],
                                sha256=target['row']['sha256'], first=target['first'],
                                array_digest=target['row']['array_digest'], exact=target['exact']))
    dynamic = None
    if stage == 'iterative_recovery':
        dynamic = [dict(arrays=target_arrays, first=target['first'],
                        id='engineering_dynamic_fixture_reuses_existing_verified_recovery')]
    g = L.groups(q, pp, anchors, targets, additional_targets=dynamic)
    assert E.digest(q.state_dict()) == before_audit
    opt = L.make_optimizer(q, pp)
    permitted, events, calls = [opt], [], 0
    maximum_calls = 6 if stage == 'iterative_recovery' else 3
    original_step = torch.optim.Adam.step
    runtime = _runtime()

    def step(optimizer, *args, **kwargs):
        nonlocal calls
        guard()
        assert any(optimizer is value for value in permitted)
        assert calls < maximum_calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        return result

    def event(which, model, optimizer):
        def emit(where, k):
            guard()
            assert where in ['BEFORE', 'AFTER'] and k == 0
            gradient = {name: value.grad for name, value in model.named_parameters()
                        if value.requires_grad}
            assert all(value is not None for value in gradient.values())
            events.append(dict(which=which, where=where, k=k,
                               hashes=E.L.statehash(model, optimizer),
                               gradient_sha256=E.digest(gradient),
                               gradient_dtypes={name: str(value.dtype) for name, value in gradient.items()}))
        return emit

    torch.optim.Adam.step = step
    try:
        transactions = [_transaction(q, opt, g, pp, 'PPO', anchors[1][0], guard, permitted, event)]
        if stage == 'iterative_recovery':
            bc_q = _frozen(pp)
            bc_opt = L.make_optimizer(bc_q, pp)
            permitted.append(bc_opt)
            transactions.append(_transaction(bc_q, bc_opt, g, pp, 'BC', anchors[1][0], guard, permitted, event))
        assert calls == maximum_calls
        for kind in ['PPO'] + (['BC'] if stage == 'iterative_recovery' else []):
            live = [{k: v for k, v in r.items() if k != 'which'} for r in events
                    if r['which'] == kind + '_mature_live']
            shadow = [{k: v for k, v in r.items() if k != 'which'} for r in events
                      if r['which'] == kind + '_mature_shadow']
            assert len(live) == len(shadow) == 2 and live == shadow
    finally:
        torch.optim.Adam.step = original_step
    contract = dict(protocol_sha256=E.sha(pfile), relay_stage=stage,
                    sources={str(f): E.sha(f) for f in [Path(__file__).resolve(),
                              ROOT/'stage/experiments/t439_representation_rl_control/t439_learning.py']},
                    starting_model_sha256=p['student_start_sha256'],
                    starting_policy_hash=p['expected_policy_hash'],
                    fixture_sources=source_receipts, trainable_parameters=L.trainable_count(q),
                    trainable_prefixes=L.trainable_prefixes(q),
                    runtime_before=runtime, runtime_after=_runtime(),
                    transactions=transactions, full_gradient_and_state_events=events,
                    mature_live_shadow_gradients_Actor_Adam_exact=True,
                    rejection_and_interruption_restore_exact=True,
                    complete_observation_history_not_reset=True,
                    native=0, new_navigation_training=0, retained_Actor=0,
                    discarded_Adam_total=calls, no_model_export=True,
                    iid_tape_is_synthetic_engineering_data=True,
                    synthetic_actions_did_not_generate_observations=True,
                    synthetic_rewards_not_physics_or_reward_validation=True,
                    not_onpolicy_navigation_data=True,
                    dynamic_fixture_reuses_existing_recovery_not_new_data=stage == 'iterative_recovery')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    E.save(outfile, dict(state='PASS_DISCARDED_ADAM_ENGINEERING_CONTRACT',
                         stage=stage, label=label, pid=os.getpid(),
                         elapsed_s=time.time()-started, contract=contract))
    print('T439_CLONE_PASS', stage, label, 'discarded_Adam', calls, flush=True)


def compare(pfile, stage, afile, bfile, outfile):
    assert stage in STAGES
    outfile = Path(outfile).resolve()
    assert not outfile.exists()
    a, b = E.read(afile), E.read(bfile)
    assert a['state'] == b['state'] == 'PASS_DISCARDED_ADAM_ENGINEERING_CONTRACT'
    assert a['stage'] == b['stage'] == stage
    assert a['label'] == 'A' and b['label'] == 'B' and a['pid'] != b['pid']
    assert a['contract'] == b['contract'], 'Independent process contracts differ'
    c = a['contract']
    assert c['protocol_sha256'] == E.sha(pfile)
    assert c['discarded_Adam_total'] == (6 if stage == 'iterative_recovery' else 3)
    E.save(outfile, dict(state='PASS', stage=stage, processes_distinct=True,
                         full_numeric_contract_exact=True,
                         discarded_Adam_total=2*c['discarded_Adam_total'],
                         native=0, new_navigation_training=0, retained_Actor=0,
                         synthetic_engineering_not_navigation=True,
                         clone_A_sha256=E.sha(afile), clone_B_sha256=E.sha(bfile)))
    print('T439_CLONE_COMPARE_PASS', stage, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'probe':
        assert len(sys.argv) == 6
        probe(*sys.argv[2:])
    else:
        assert sys.argv[1] == 'compare' and len(sys.argv) == 7
        compare(*sys.argv[2:])
