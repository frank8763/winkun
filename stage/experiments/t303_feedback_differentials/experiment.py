"""Fixed historical differential diagnostic; never a new Actor training run.

The only Adam calls reconstruct T302's already rejected first block. No native
environment, policy promotion, new rollout, or Jacobian regularizer is used.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t303_feedback_differentials'
sys.path.insert(0, str(ROOT / 'stage/experiments/t302_balanced_bridge_learning'))
import t302_learning as T
import argparse, ast, copy, fcntl, gzip, io, json, os, shutil, signal, subprocess, time, traceback
np, torch, C, S = T.np, T.torch, T.C, T.S


def read(path):
    return json.loads(Path(path).read_text())


def payload(path):
    with np.load(path) as z:
        a = {k: z[k].copy() for k in z.files}
    if 'last_next_state' in a:
        a = T.unpack_full(a)
    assert a['states'].shape[1] == a['next_states'].shape[1] == 120
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    return a


def resources(p):
    assert S.used(OUT) <= p['final_bytes']
    assert S.used(ROOT / 'stage/runs') <= p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for path, digest in p['input_hashes'].items():
        assert C.sha(path) == digest, path
    resources(p)


def prepare():
    assert not OUT.exists()
    basepath = ROOT / 'stage/runs/t302_balanced_bridge_learning/protocol.json'
    summarypath = basepath.parent / 'summary.json'
    base, summary = read(basepath), read(summarypath)
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped']
    block = summary['supervised']['blocks'][0]
    assert len(summary['supervised']['blocks']) == 1 and not block['accepted']
    assert block['steps_attempted'] == summary['supervised']['attempted_steps'] == 50
    assert summary['supervised']['retained_steps'] == 0
    inputs = dict(base['input_hashes'])
    casespath = ROOT / 'stage/runs/t284_feedback_drift/cases.json'
    row = next(r for r in read(casespath) if r['id'] == 'parent')
    parentpath = T.resolve(row['path'])
    assert C.sha(parentpath) == row['sha256']
    assert row['policy_hash'] == base['policy_hash']
    sources = [dict(id='old_LEFT5m_dev24', path=str(parentpath), sha256=row['sha256'], family='old_critical')]
    sources += [dict(id=s['id'], path=s['path'], sha256=s['sha256'], family=s['side'] + '_own_success') for s in base['auxiliary_sources']]
    assert len(sources) == 16
    for source in sources:
        assert C.sha(source['path']) == source['sha256']
        a = payload(source['path'])
        source['indices'] = np.linspace(0, len(a['states']) - 1, 4).astype(int).tolist()
        source['states_sha256'] = S.digest(a['states'])
        inputs[source['path']] = source['sha256']
    for f in [Path(__file__), basepath, summarypath, casespath,
              ROOT / 'stage/docs/distance_20261102/T302_AUDIT.json',
              ROOT / 'stage/docs/distance_20261102/LITERATURE_AFTER_T302.md',
              ROOT / 'stage/docs/distance_20261102/T303_PLAN.md']:
        inputs[str(f)] = C.sha(f)
    p = dict(experiment='T303', source_protocol=str(basepath), source_summary=str(summarypath),
             sources=sources, input_hashes=inputs,
             parent_policy_hash=base['policy_hash'], expected_proposal=block,
             reconstruction_steps=50, audit_reconstruction_steps=50, preflight_optimizer_calls=0,
             max_commands=0, max_physics=0, new_navigation_Actor_updates=0,
             wall_seconds=600, audit_wall_seconds=600, final_bytes=2 * 1024**2,
             global_bytes=1024**3, min_free_bytes=40 * 1024**3,
             sample_count=64, payload_limit_bytes=640 * 1024,
             finite_difference_epsilon=1e-5,
             contract='Fixed descriptive offline pilot, not new Actor learning, not a TaSIL implementation or stability certificate. Reconstruct the exact rejected T302 proposal with 50 disposable historical Adam calls and independent 50-call audit; no continuation/extra steps/seed/grid. Parent and proposal same canonical batch1 hidden-reset Actor. Exactly four uniform decision starts per 16 already consumed full120 trajectories: one safe oldLEFT5m24, five actual LEFT5.5m and ten RIGHT6m successes. Hold06 full120 unavailable here and not fabricated. Measure mode/conditional marginal center/raw outputs and their 2x120 input Jacobians, signed raw clipping margins, and directional derivatives along recorded next120-state differences. Mixed physical input units; Jacobian norms are coordinate-dependent. Recorded successor direction includes time/goal history; interpolated inputs for numerical finite difference are not claimed physically attainable. Float64 cloned-policy finite differences validate local differentiation only, not exact float32 native control. No threshold turns these descriptive statistics into causal root cause, numerical global stability, successful navigation or automatic permission for Jacobian regularization. No native environments, commands, new behavior, Critic or regularizer. Final scientific decision requires existing closed-loop evidence and a separately bounded matched intervention. No Phase/map/planner/waypoint/teacher/router; no6m promotion or7/8/global launch.')
    assert S.used(ROOT / 'stage/runs') + p['final_bytes'] < p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']
    ast.parse(Path(__file__).read_text())
    OUT.mkdir()
    for path in [HERE / 'protocol.json', OUT / 'protocol.json']:
        C.save(path, p)
    print('T303 registered: 0 native; historical reconstruction only', flush=True)


def selected(p):
    xx, nn, sources, indices = [], [], [], []
    for i, source in enumerate(p['sources']):
        a = payload(source['path'])
        assert S.digest(a['states']) == source['states_sha256']
        ix = source['indices']
        xx.extend(a['states'][ix]); nn.extend(a['next_states'][ix])
        sources.extend([i] * len(ix)); indices.extend(ix)
    return dict(states=np.asarray(xx), next_states=np.asarray(nn),
                source_index=np.asarray(sources, dtype=np.int32),
                decision_index=np.asarray(indices, dtype=np.int32))


def differential(actor, state):
    x = torch.tensor(state[None], requires_grad=True)
    outputs = actor.step(x)
    fields = [outputs[k][0] for k in [0, 1, 3]]
    js = []
    for f in fields:
        js.append(torch.stack([torch.autograd.grad(f[k], x, retain_graph=True)[0][0] for k in range(2)]))
    return torch.stack(fields).detach().numpy(), torch.stack(js).detach().numpy()


def numerical_validation(actor, state, direction, epsilon):
    x = torch.tensor(state[None], dtype=torch.float64, requires_grad=True)
    d = torch.tensor(direction[None], dtype=torch.float64)
    out = actor.step(x)
    ys = torch.stack([out[k][0] for k in [0, 1, 3]])
    exact = torch.stack([torch.stack([(torch.autograd.grad(y[k], x, retain_graph=True)[0] * d).sum() for k in range(2)]) for y in ys])
    with torch.no_grad():
        plus, minus = actor.step(x + epsilon * d), actor.step(x - epsilon * d)
        finite = torch.stack([(plus[k][0] - minus[k][0]) / (2 * epsilon) for k in [0, 1, 3]])
    return exact.detach().numpy(), finite.numpy()


def preflight():
    p = read(HERE / 'protocol.json')
    assert not any((OUT / name).exists() for name in ['launch.json', 'status.json', 'summary.json'])
    verify(p)
    data = selected(p)
    assert data['states'].shape == data['next_states'].shape == (64, 120)
    # Analytical linear map independently checks differentiation axes, no Adam/native.
    x = torch.arange(120., dtype=torch.float64, requires_grad=True)
    w = torch.arange(240., dtype=torch.float64).reshape(2, 120) / 240
    jac = torch.autograd.functional.jacobian(lambda v: w @ v, x)
    assert torch.equal(jac, w)
    full = S.used(ROOT / 'stage/runs') - S.used(OUT) + p['final_bytes']
    assert full < p['global_bytes']
    C.save(OUT / 'preflight.json', dict(state='PASS', native_steps=0, optimizer_calls=0,
           native_environments_created=0, sample120_hash=S.digest(data['states']),
           full_reserved_global_bytes=full, protocol_sha256=C.sha(HERE / 'protocol.json')))
    print('T303 zero-native/zero-optimizer preflight PASS', flush=True)


def reconstruct(p, guard, callback):
    base = read(p['source_protocol'])
    q = T.policy(base)
    frozen = copy.deepcopy(q).eval().requires_grad_(False)
    x, y, offsets = T.anchors(base, frozen)
    items = T.auxiliary(base)
    opt = T.L.optimizer(q, base)
    events = []
    for step in range(p['reconstruction_steps']):
        guard()
        callback('started', step, events)
        event = T.supervised_step(q, opt, items, x, y, offsets, base, step)
        events.append(event)
        callback('returned', step, events)
    assert events[-1] == p['expected_proposal']['last_step']
    assert S.digest(q.state_dict()) == p['expected_proposal']['proposal_policy_hash']
    assert S.digest(opt.state_dict()) == p['expected_proposal']['proposal_optimizer_hash']
    return frozen, q.eval().requires_grad_(False), events


def calculate(p, guard, callback, reverse=False):
    parent, proposal, events = reconstruct(p, guard, callback)
    data = selected(p)
    model_values, model_jacobians, numerical = [], [], []
    for actor in [parent, proposal]:
        values, jacobians, checks = {}, {}, {}
        double = copy.deepcopy(actor).double()
        for i in (reversed(range(64)) if reverse else range(64)):
            guard()
            state = data['states'][i]
            direction = (data['next_states'][i] - state).astype(np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            values[i], jacobians[i] = differential(actor, state)
            exact, finite = numerical_validation(double, state, direction, p['finite_difference_epsilon'])
            checks[i] = np.stack([exact, finite])
        model_values.append(np.stack([values[i] for i in range(64)]))
        model_jacobians.append(np.stack([jacobians[i] for i in range(64)]))
        numerical.append(np.stack([checks[i] for i in range(64)]))
    arrays = dict(data, values=np.stack(model_values), input_jacobians=np.stack(model_jacobians),
                  directional_validation_float64=np.stack(numerical))
    assert all(np.isfinite(a).all() for a in arrays.values())
    rows = []
    for j, source in enumerate(p['sources']):
        ix = np.flatnonzero(data['source_index'] == j)
        v, g = arrays['values'][:, ix], arrays['input_jacobians'][:, ix]
        fields = {}
        for k, name in enumerate(['deployed_mode', 'marginal_center', 'preclip_raw']):
            delta = v[1, :, k] - v[0, :, k]
            dg = g[1, :, k] - g[0, :, k]
            parentnorm = float(np.linalg.norm(g[0, :, k]))
            fields[name] = dict(value_RMS=float(np.sqrt(np.mean(delta**2))),
                value_max_abs=float(np.abs(delta).max()),
                jacobian_difference_norm=float(np.linalg.norm(dg)),
                parent_jacobian_norm=parentnorm,
                jacobian_relative_difference=float(np.linalg.norm(dg) / max(parentnorm, 1e-12)))
        raw = v[:, :, 2]
        clipped = np.abs(raw) >= .999999
        rows.append(dict(id=source['id'], family=source['family'], indices=source['indices'], fields=fields,
                         raw_signed_clipping_margin=(.999999 - np.abs(raw)).tolist(),
                         clipping_status_changes=int(np.sum(clipped[0] != clipped[1])),
                         parent_saturated_components=int(clipped[0].sum()),
                         proposal_saturated_components=int(clipped[1].sum())))
    n = arrays['directional_validation_float64']
    numerical_error = np.abs(n[:, :, 0] - n[:, :, 1])
    report = dict(rows=rows, field_order=['mode', 'marginal_center', 'raw'], model_order=['safeT255', 'historical_T302_rejected'],
                  sample_count=64, sources=16, numerical_validation_absolute_max_by_field=numerical_error.max(axis=(0, 1, 3)).tolist(),
                  numerical_validation_relative_L2_by_field=[float(np.linalg.norm(numerical_error[:, :, k]) / max(np.linalg.norm(n[:, :, 0, k]), 1e-12)) for k in range(3)],
                  proposal_policy_hash=p['expected_proposal']['proposal_policy_hash'],
                  proposal_optimizer_hash=p['expected_proposal']['proposal_optimizer_hash'],
                  parent_policy_hash=S.digest(parent.state_dict()), disposable_historical_Adam_calls=len(events),
                  caveat=p['contract'])
    return arrays, report, events


def run():
    p = read(HERE / 'protocol.json')
    assert not (OUT / 'status.json').exists()
    start = time.time()
    def guard():
        assert time.time() < start + p['wall_seconds'] - 10
        resources(p)
    def alarm(*_):
        raise TimeoutError('T303 absolute diagnostic wall deadline')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(p['wall_seconds'] - 5)
    lock = (ROOT / 'stage/runs/.stage_native_exclusive.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary = dict(state='FAILED', pid=os.getpid(), native_steps=0, physics_steps=0, commands=0,
                   Actor_updates=0, new_navigation_training_updates=0, all_workers_stopped=True)
    C.save(OUT / 'workers.json', dict(parent=os.getpid(), workers=[]))
    def callback(state, step, events):
        C.save(OUT / 'optimizer_calls.json', dict(disposable_historical_calls_returned=len(events),
               inflight_historical_step=step if state == 'started' else None, new_navigation_updates=0))
        if state == 'returned' and (step + 1) % 10 == 0:
            C.save(OUT / 'status.json', dict(state='HISTORICAL_RECONSTRUCTION', pid=os.getpid(),
                   updated_epoch=time.time(), absolute_deadline_epoch=start+p['wall_seconds'], historical_steps=len(events)))
    try:
        verify(p)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        C.save(OUT / 'status.json', dict(state='STARTED', pid=os.getpid(), updated_epoch=time.time(), absolute_deadline_epoch=start+p['wall_seconds']))
        arrays, report, events = calculate(p, guard, callback)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        blob = buf.getvalue()
        assert len(blob) <= p['payload_limit_bytes']
        (OUT / 'differentials.npz').write_bytes(blob)
        C.save(OUT / 'analysis.json', report)
        C.save(OUT / 'historical_steps.json', events)
        verify(p)
        summary.update(state='COMPLETE', historical_reconstruction_Adam_calls=len(events),
                       payload_sha256=C.sha(OUT / 'differentials.npz'), array_sha256={k:S.digest(a) for k,a in arrays.items()},
                       proposal_policy_hash=report['proposal_policy_hash'], proposal_optimizer_hash=report['proposal_optimizer_hash'])
    except BaseException:
        C.save(OUT / 'failure.json', dict(traceback=traceback.format_exc(), no_navigation=True))
    finally:
        signal.alarm(0)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    summary.update(main_epoch=start, elapsed_s=time.time()-start, absolute_deadline_epoch=start+p['wall_seconds'], own_bytes=S.used(OUT), global_bytes=S.used(ROOT/'stage/runs'))
    C.save(OUT / 'summary.json', summary)
    C.save(OUT / 'status.json', summary)
    resources(p)
    print(json.dumps(summary), flush=True)
    if summary['state'] != 'COMPLETE':
        raise SystemExit(1)


def stopped():
    summary = read(OUT / 'summary.json')
    for pid in [summary['pid'], read(OUT / 'launch.json')['pid']]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(('Live or reused process', pid))
    return summary


def audit():
    dest = ROOT / 'stage/docs/distance_20261102/T303_AUDIT.json'
    assert not dest.exists()
    p, summary = read(HERE / 'protocol.json'), stopped()
    assert summary['state'] == 'COMPLETE'
    verify(p)
    start = time.time()
    def guard():
        assert time.time() < start + p['audit_wall_seconds']
        resources(p)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays, report, events = calculate(p, guard, lambda *args: None, reverse=True)
    with np.load(OUT / 'differentials.npz') as z:
        assert set(z.files) == set(arrays)
        for k, a in arrays.items():
            assert np.array_equal(a, z[k]) and S.digest(a) == summary['array_sha256'][k], k
    assert report == read(OUT / 'analysis.json') and events == read(OUT / 'historical_steps.json')
    assert C.sha(OUT / 'differentials.npz') == summary['payload_sha256']
    stopped(); verify(p)
    C.save(dest, dict(state='PASS', native_steps=0, new_navigation_Actor_updates=0,
           disposable_historical_Adam_reconstruction_calls=len(events), all_sample_arrays_Jacobians_and_Adam_hashes_exact=True,
           reversed_sample_order_same_hidden_reset_results=True, all_original_processes_absent=True,
           numerical_validation=report['numerical_validation_absolute_max_by_field'],
           numerical_validation_is_not_closedloop_safety=True, summary_sha256=C.sha(OUT/'summary.json'),
           source_sha256=C.sha(__file__), elapsed_s=time.time()-start, global_bytes=S.used(ROOT/'stage/runs')))
    print('T303 independent historical reconstruction/differential audit PASS', flush=True)


def launch():
    assert not any((OUT / name).exists() for name in ['launch.json', 'status.json', 'summary.json'])
    assert read(OUT / 'preflight.json')['state'] == 'PASS'
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip()
    receipt = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git','rev-parse','HEAD^{tree}'], cwd=ROOT, text=True).strip()
    p = read(HERE / 'protocol.json'); verify(p)
    with (OUT / 'run.log').open('xb') as log:
        proc = subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=1s',str(p['wall_seconds'])+'s',
               str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    C.save(OUT / 'launch.json', dict(pid=proc.pid, launch_epoch=time.time(), single_launch=True,
           source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
           remote_commit=receipt['remote_commit'], wall_seconds=p['wall_seconds']))
    print(json.dumps(dict(pid=proc.pid)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','preflight','launch','run','audit'])
    args = parser.parse_args()
    globals()[args.mode]()
