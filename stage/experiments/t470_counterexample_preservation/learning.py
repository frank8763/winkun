"""Finite joint repair with explicit off-policy function preservation.

Rejected A325 histories supply observation support, never action targets,
safe own-policy examples, PPO data, Actor weights, or Adam state. All eight
extra targets are the original input327 unclipped sequence location from h0.
This fixed-support joint pilot is not a causal or algorithmic novelty claim.
Import defines functions only; registration and the audited T469 recovery
gate belong to the caller, before any student or optimizer is constructed.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'stage470_pinned468_learning',
    ROOT/'stage/experiments/t468_offline_recovery_night/night_learning.py')
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)
V, Y, L, X, J, H, M, P, E, D, F, R = (
    N.V, N.Y, N.L, N.X, N.J, N.H, N.M, N.P, N.E, N.D, N.F, N.R)
torch, np = N.torch, N.np
read, save, sha, digest = N.read, N.save, N.sha, N.digest
state, load, export, frozen, base_digest = (
    N.state, N.load, N.export, N.frozen, N.base_digest)
MODES = ('occupancy',)
COUNTEREXAMPLE_IDS = {
    'dev_task6_03', 'dev_task6_15',
    'wide117_80_dev6_LEFT_15', 'wide117_80_dev6_LEFT_17'}


def _input_model(p):
    initial, rejected = p['learning_initial'], p['counterexample_model']
    assert initial['policy_hash'] != rejected['policy_hash']
    assert initial['sha256'] != rejected['sha256']
    assert sha(initial['path']) == initial['sha256']
    return initial, rejected


def _parameters(q):
    assert sum(v.numel() for v in q.parameters() if v.requires_grad) == 1074
    assert sum(v.numel() for v in q.parameters() if not v.requires_grad) == 52866
    L._assert_frozen(q)


def make(p, device='cuda'):
    _input_model(p)
    # N.make restores only the input327 Actor and its mature residual Adam.
    obj = N.make(p, device)
    _parameters(obj[0])
    return obj


def _history(row, rejected, recovery=False):
    a = N._arrays(row, safe=recovery)
    n = len(a['states'])
    assert n > 0 and a['states'].shape == a['next_states'].shape == (n, 120)
    assert row['policy_hash'] == rejected['policy_hash']
    assert row['loaded_model_sha256'] == rejected['sha256']
    assert a['actions'].shape == a['locations'].shape == (n, 2)
    assert a['hidden_before'].shape == (n, 64)
    assert a['last_hidden_after'].shape == (64,)
    assert np.array_equal(a['hidden_before'][0], np.zeros(64))
    assert a['history_poses'].shape == (n+1, 4, 3)
    assert a['history_goals'].shape == (n+1, 4, 2)
    assert a['history_times'].shape == (n+1, 4)
    assert a['ownership'].shape == a['stochastic'].shape == (n,)
    assert a['end_flags'].shape == (n, 3) and not a['stochastic'].any()
    counts = a['native_counts']
    assert counts.shape == (n,) and np.issubdtype(counts.dtype, np.integer)
    assert (counts > 0).all() and int(counts.sum()) == row['physics']
    assert digest(a['final_native_integration']) == row['integration_sha256']
    for i, x in enumerate(a['states']):
        rebuilt = E.N.K.H.rebuild(x, a['poses'][i], a['history_poses'][i],
                                a['task_goal_xy'], a['history_goals'][i])
        assert np.abs(rebuilt-x).max() <= 2e-6
    if recovery:
        first = row['teacher_first']
        assert row['teacher'] and isinstance(first, int) and 0 <= first < n
        assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert row['teacher_kind'] == 'T463_PPO327_actual_history_carry'
        assert row['teacher_hidden_burns_actual_history_from_h0']
        assert a['teacher_actions'].shape == (n, 2)
    else:
        assert not row['teacher'] and not row['success']
        assert not a['ownership'].any()
    return a


def _target(anchor_cpu, anchor, states, device):
    """Check every complete-history location; no clipped action is a target."""
    x_cpu = torch.tensor(states, device='cpu')
    x = torch.tensor(states, device=device)
    with torch.no_grad():
        cpu = anchor_cpu.sequence(x_cpu).detach()
        anchor_cpu.reset_noise()
        hidden, steps = None, []
        for value in x_cpu:
            _, location, hidden, _ = anchor_cpu.step(value[None], hidden)
            steps.append(location[0].detach())
        step = torch.stack(steps)
        assert cpu.shape == step.shape == (len(states), 2)
        step_error = float((cpu-step).abs().max())
        assert step_error < 1e-10
        anchor_cpu.reset_noise()
        target = anchor.sequence(x).detach().clone()
        assert target.shape == cpu.shape and torch.isfinite(target).all()
        device_error = float((target.cpu()-cpu).abs().max())
        # Engineering float64 device comparison only; no native action or
        # physical reproduction threshold is changed by this check.
        assert device_error < 1e-8
        initial_error = float((anchor.sequence(x)-target).square().mean())
        assert initial_error == 0.
    return x, target, dict(cpu_step_sequence_max_abs=step_error,
        cpu_device_sequence_max_abs=device_error,
        cpu_target_digest=digest(cpu), device_target_digest=digest(target),
        initial_counterexample_function=initial_error)


def data(p, device='cuda'):
    initial, rejected = _input_model(p)
    students, recoveries = (p['counterexample_student_rows'],
                            p['counterexample_recovery_rows'])
    assert len(students) == len(recoveries) == 4
    assert {r['job_id'] for r in students} == COUNTEREXAMPLE_IDS
    assert {r['job_id'] for r in recoveries} == COUNTEREXAMPLE_IDS
    # Preserve T468's two corrections, valid current-success references and
    # genuine safe-own support without passing any rejected-A rows into it.
    batch = N.data(p, p['seed_occupancy_rows'], p['validated_corrections'],
                   'occupancy', device=device)
    assert batch['correction_count'] == 2
    assert {g['id']: g['first'] for g in batch['corrections']} == {
        'dev_task6_00': 26, 'dev_task6_21': 20}
    assert not ({r['path'] for r in students+recoveries} &
                set(batch['current_own_support_paths']))
    source = {r['job_id']: _history(r, rejected) for r in students}
    anchor_cpu = frozen(p, initial['path']).cpu().eval().requires_grad_(False)
    anchor = frozen(p, initial['path']).to(device).eval().requires_grad_(False)
    assert anchor_cpu.phash() == anchor.phash() == initial['policy_hash']
    groups, provenance = [], []
    for kind, rows in [('rejected_student', students), ('validated_recovery', recoveries)]:
        for row in rows:
            a = source[row['job_id']] if kind == 'rejected_student' else _history(row, rejected, True)
            if kind == 'validated_recovery':
                assert row['teacher_source_SHA256'] == initial['sha256']
                first, original = row['teacher_first'], source[row['job_id']]
                assert first < len(original['states'])
                for key in ['states', 'poses', 'hidden_before']:
                    assert np.array_equal(a[key][:first+1], original[key][:first+1]), key
                for key in ['actions', 'native_counts', 'next_states', 'end_flags']:
                    assert np.array_equal(a[key][:first], original[key][:first]), key
                for key in ['history_poses', 'history_goals', 'history_times']:
                    assert np.array_equal(a[key][:first+1], original[key][:first+1]), key
            x, target, check = _target(anchor_cpu, anchor, a['states'], device)
            groups.append(dict(id=row['job_id'], kind=kind, x=x, target=target))
            provenance.append(dict(id=row['job_id'], kind=kind, path=row['path'],
                sha256=row['sha256'], array_digest=row['array_digest'],
                source_policy_hash=row['policy_hash'], target_policy_hash=initial['policy_hash'],
                target_model_sha256=initial['sha256'], decisions=len(x),
                physical_steps=row['physics'], equal_episode_weight=1/8,
                target='input327.sequence(full_actual_history_from_h0)_unclipped_location',
                offpolicy_REJECTED_diagnostic_model_NOT_newonpolicy=True,
                source_actions_are_NOT_training_targets=True, **check))
    assert len(groups) == 8
    batch.update(counterexample_groups=groups, counterexample_count=8,
        counterexample_provenance=provenance, counterexample_total_weight=1.,
        initial_counterexample_function=0.,
        offpolicy_REJECTED_diagnostic_model_NOT_newonpolicy=True)
    return batch


def bc_step(obj, batch, mode=None):
    mode = batch['mode'] if mode is None else mode
    assert mode == batch['mode'] == 'occupancy'
    q, opt = obj
    assert q._t463_mode == 'aggregate'
    _parameters(q)
    opt.zero_grad(set_to_none=True)
    result = dict(mode=mode, student_prefix=0., teacher_suffix=0., duration_loss=0.,
        correction_objective=0., reference218=0., own10=0., counterexample_function=0.)
    count = len(batch['corrections'])
    assert count == 2
    for g in batch['corrections']:
        loc = q.sequence(g['x'])
        loss_per_axis = P.action_loss(loc, g['y'])
        n = g['prefix_decisions']
        early = loss_per_axis[:n].mean()
        suffix = loss_per_axis[n:].mean() if n < len(loc) else loc.sum()*0.
        duration = (loss_per_axis.mean(dim=1)*g['native_weight']).sum()
        loss = early+.1*suffix
        (loss/count).backward()
        result['student_prefix'] += float(early.detach())/count
        result['teacher_suffix'] += float(suffix.detach())/count
        result['duration_loss'] += float(duration.detach())/count
        result['correction_objective'] += float(loss.detach())/count
    result['reference218'], result['own10'] = N._retention(q, batch)
    groups = batch['counterexample_groups']
    assert len(groups) == batch['counterexample_count'] == 8
    assert batch['counterexample_total_weight'] == 1.
    for g in groups:
        loc = q.sequence(g['x'])
        assert loc.shape == g['target'].shape
        loss = (loc-g['target']).square().mean()/len(groups)
        loss.backward()
        result['counterexample_function'] += float(loss.detach())
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
        L._trainable(q), 1., error_if_nonfinite=True))
    _parameters(q)
    opt.step()
    _parameters(q)
    return result
