"""Fixed-prefix, RAM-only reconstruction; never generates navigation data."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t264_offline_gradient_audit'
sys.path.insert(0, str(ROOT / 'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L, S, np, torch, policy, anchors, closed_loop_update


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    limit = json.loads((HERE / 'protocol.json').read_text())['max_output_bytes']
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    prior = (OUT / name).stat().st_size if (OUT / name).exists() else 0
    assert used - prior + len(data) < limit
    assert sum(p.stat().st_size for p in (ROOT / 'stage/runs').rglob('*') if p.is_file()) + len(data) < 1024**3
    assert shutil.disk_usage(ROOT).free >= 40 * 1024**3
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_bytes(data)


def load_fixed(plan):
    for name, expected in plan['input_hashes'].items():
        assert sha(Path(name).read_bytes()) == expected, name
    fixed = {}
    for key, count in [('training', 48), ('updates', 2)]:
        lines = Path(plan[key + '_log']).read_bytes().splitlines(keepends=True)[:count]
        assert len(lines) == count and sha(b''.join(lines)) == plan[key + '_prefix_sha256']
        fixed[key] = [json.loads(line) for line in lines]
    items = []
    for row in fixed['training']:
        assert row['complete'] and sha(Path(row['path']).read_bytes()) == row['sha256']
        with np.load(row['path']) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert arrays['states'].shape == arrays['next_states'].shape
        assert arrays['states'].shape[1] == 120
        assert all(S.digest(arrays[k]) == h for k, h in row['array_hashes'].items())
        items.append(dict(row=row, arrays=arrays))
    fixed['items'] = items
    return fixed


def forbid(*args, **kwargs):
    raise AssertionError('T264 forbids simulation, workers, and new navigation data')


def objective(p, data, plan):
    lp, _ = L.likelihood(p, data)
    delta = lp - data['oldlp']
    ratio = delta.exp()
    advantage = data['advantage']
    term = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage) * data['weights']
    scores = [term[data['task'] == t].sum() / 4 / plan['surrogate_horizon_decisions'] for t in range(4)]
    return scores, term


def diagnostics(p, data, items, ax, ay, offset, plan, rnd):
    params = [q for q in p.parameters() if q.requires_grad]
    flat = lambda g: torch.cat([v.reshape(-1) for v in g])
    scores, _ = objective(p, data, plan)
    gs = [flat(torch.autograd.grad(-s, params, retain_graph=True)) for s in scores]
    rng = np.random.default_rng(plan['seed'] + rnd)
    indices = np.concatenate([rng.integers(offset[t], offset[t+1], 32) for t in range(8)])
    ret = plan['retention_weight'] * (p.step(ax[indices])[0] - ay[indices]).square().mean()
    rg = flat(torch.autograd.grad(ret, params))
    pg, conflicts = L.project(gs, plan['seed'] + 100 * rnd)
    average = torch.stack(gs).mean(0)
    total = pg + rg
    cosine = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    ids = sorted({i['row']['source_reset_id'] for i in items})
    reports = []
    for task, name in enumerate(ids):
        rr = [i for i in items if i['row']['source_reset_id'] == name]
        mc = [S.monte_carlo(i['arrays']['rewards'], i['arrays']['native_counts']) for i in rr]
        reports.append(dict(id=name, side=rr[0]['row']['job']['case']['side'],
            safe=sum(i['row']['success'] and not i['row']['collision'] for i in rr),
            contacts=sum(i['row']['collision'] for i in rr), timeouts=sum(i['row']['timeout'] for i in rr),
            returns_at_start=[float(x[0]) for x in mc],
            initial_leave_one_out_advantages=[float(x[0] - np.mean([v[0] for k,v in enumerate(mc) if k != j])) for j,x in enumerate(mc)],
            loss_gradient_norm=float(gs[task].norm()),
            cosine_loss_gradient_with_PCGrad=cosine(gs[task], pg),
            cosine_loss_gradient_with_soft_retention=cosine(gs[task], rg),
            cosine_loss_gradient_with_total=cosine(gs[task], total),
            unit_SGD_first_order_surrogate_gain=float(torch.dot(gs[task], total))))
    return dict(round=rnd, policy_hash=S.digest(p.state_dict()), tasks=reports,
        pairwise_loss_gradient_cosines=[[cosine(a,b) for b in gs] for a in gs],
        plain_average_gradient_norm=float(average.norm()), PCGrad_norm=float(pg.norm()),
        soft_retention_gradient_norm=float(rg.norm()), total_gradient_norm=float(total.norm()),
        retention_to_PCGrad_norm_ratio=float(rg.norm()/pg.norm().clamp_min(1e-30)),
        PCGrad_retained_norm_vs_average=float(pg.norm()/average.norm().clamp_min(1e-30)),
        directed_projection_conflicts=conflicts,
        note='Local sampled stochastic surrogate derivatives; SGD signs are not Adam effects, deterministic navigation, or causal proof.')


def main(preflight=False):
    started = time.monotonic()
    plan = json.loads((HERE / 'protocol.json').read_text())
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T264 fixed wall limit')))
    signal.alarm(plan['wall_seconds'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    S.native_setup = forbid
    L.generate = forbid
    assert not (OUT / ('preflight.json' if preflight else 'status.json')).exists(), 'No duplicate audit run'
    fixed = load_fixed(plan)
    training_plan = json.loads(Path(plan['t263_protocol']).read_text())
    p = policy(training_plan)
    data = L.training_batch(fixed['items'][:16], p)
    assert S.digest(p.state_dict()) == fixed['training'][0]['policy_hash']
    if preflight:
        write('preflight.json', dict(state='PASS', physics_steps=0, optimizer_calls=0,
            fixed_payloads_verified=48, initial_behavior_density=data['density'],
            simulator_entrypoints_disabled=True, protocol_sha256=sha((HERE/'protocol.json').read_bytes())))
        return
    assert json.loads((OUT/'preflight.json').read_text())['state'] == 'PASS'
    write('status.json', dict(state='RUNNING', pid=os.getpid(), physics_steps=0,
                             clone_optimizer_calls=0, navigation_RL_updates=0))
    clone_calls = 0
    original_step = torch.optim.Adam.step
    def counted_step(opt, *args, **kwargs):
        nonlocal clone_calls
        clone_calls += 1
        assert clone_calls <= 8
        return original_step(opt, *args, **kwargs)
    torch.optim.Adam.step = counted_step
    try:
        frozen = copy.deepcopy(p).eval().requires_grad_(False)
        ax, ay, offset = anchors(training_plan, frozen)
        opt = L.optimizer(p, training_plan)
        native = {}
        for row in fixed['updates']:
            for update in row['updates']:
                for attempt in update['attempts']:
                    record = attempt.get('native_old_task_guard')
                    if record:
                        if record['policy_hash'] in native:
                            assert native[record['policy_hash']] == record
                        native[record['policy_hash']] = record
        native_calls = 0
        def historical_gate(q):
            nonlocal native_calls
            native_calls += 1
            return native[S.digest(q.state_dict())]
        derivative_reports = []
        reconstructed = []
        for rnd in range(3):
            items = fixed['items'][16*rnd:16*(rnd+1)]
            expected_policy = {i['row']['policy_hash'] for i in items}
            assert expected_policy == {S.digest(p.state_dict())}
            data = L.training_batch(items, p)
            derivative_reports.append(diagnostics(p, data, items, ax, ay, offset, training_plan, rnd))
            if rnd < 2:
                expected = fixed['updates'][rnd]['updates']
                live = closed_loop_update(p,opt,data,ax,ay,offset,training_plan,rnd,
                    lambda: None,historical_gate,lambda event: None)
                assert live == expected, 'Offline clone failed exact recorded proposal/policy/Adam reconstruction'
                assert S.digest(p.state_dict()) == expected[-1]['policy_hash']
                assert S.digest(opt.state_dict()) == expected[-1]['optimizer_hash']
                reconstructed.append(dict(round=rnd, policy_hash=S.digest(p.state_dict()),
                    optimizer_hash=S.digest(opt.state_dict()), exact_all_recorded_proposal_fields=True))
            write('status.json',dict(state='RUNNING',pid=os.getpid(),physics_steps=0,
                navigation_RL_updates=0,clone_optimizer_calls=clone_calls,
                diagnosed_rounds=len(derivative_reports), elapsed_s=time.monotonic()-started))
        result = dict(state='COMPLETE', physics_steps=0, navigation_RL_updates=0,
            clone_optimizer_calls=clone_calls, historical_native_guard_lookup_calls=native_calls,
            new_native_guard_queries=0, reconstructed_rounds=reconstructed,
            round_diagnostics=derivative_reports, new_model_exported=False,
            active_T263_files_modified=False, independent_new_training_data=0,
            protocol_sha256=sha((HERE/'protocol.json').read_bytes()), elapsed_s=time.monotonic()-started,
            limitations=['Fixed 48 selected before audit; no new seed, goal or hyperparameter sweep.',
                'PCGrad cosine and norm data alone do not establish closed-loop causal effects.',
                'Historical clone Adam calls reconstruct existing updates, not new navigation RL updates.',
                'All simulated acceptance results are historical exact-policy-hash lookups, not fresh validation.',
                'No deterministic success or full old-task preservation gate is inferred from gradients.'])
        write('summary.json',result)
        write('status.json',dict(state='COMPLETE',physics_steps=0,navigation_RL_updates=0,
                                 clone_optimizer_calls=clone_calls,elapsed_s=result['elapsed_s']))
        print(json.dumps({k:result[k] for k in ['state','physics_steps','navigation_RL_updates','clone_optimizer_calls','elapsed_s']}),flush=True)
    except BaseException:
        result=dict(state='FAILED',physics_steps=0,navigation_RL_updates=0,
            clone_optimizer_calls=clone_calls,traceback=traceback.format_exc(),
            active_T263_files_modified=False, elapsed_s=time.monotonic()-started)
        write('failure.json',result)
        write('status.json',result)
        raise
    finally:
        torch.optim.Adam.step = original_step
        signal.alarm(0)


if __name__ == '__main__':
    main(preflight='--preflight' in sys.argv)
