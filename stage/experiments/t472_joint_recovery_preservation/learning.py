"""Lazy joint coverage and heterogeneous-teacher function-support pilot.

The student and mature Adam always come from input327. A325 supplies only
explicitly off-policy observation histories; its weights and failed actions
are never student initialization or own-policy targets. Registration, native
safe/exact gates, the fixed 128 BC/shadow budget and audit belong to control.
Importing this module does not construct a student, optimizer or simulator.
Finite-support preservation is neither a safety nor a causal/novelty claim.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'stage472_pinned470_learning',
    ROOT/'stage/experiments/t470_counterexample_preservation/learning.py')
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
N = C.N
V, Y, L, X, J, H, M, P, E, D, F, R = (
    C.V, C.Y, C.L, C.X, C.J, C.H, C.M, C.P, C.E, C.D, C.F, C.R)
torch, np = C.torch, C.np
read, save, sha, digest = C.read, C.save, C.sha, C.digest
state, load, export, frozen, base_digest = (
    C.state, C.load, C.export, C.frozen, C.base_digest)
MODES = ('occupancy',)
OLD_IDS = {'dev_task6_03', 'dev_task6_15'}
WIDE_IDS = {'wide117_80_dev6_LEFT_15', 'wide117_80_dev6_LEFT_17'}


def make(p, device='cuda'):
    return C.make(p, device)


def _history(row, model, safe=False):
    a = N._arrays(row, safe=safe)
    n = len(a['states'])
    assert n > 0 and a['states'].shape == a['next_states'].shape == (n, 120)
    assert row['policy_hash'] == model['policy_hash']
    assert row['loaded_model_sha256'] == model['sha256']
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
    if row['teacher']:
        first = row['teacher_first']
        assert isinstance(first, int) and 0 <= first < n
        assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert a['teacher_actions'].dtype == np.float32
        assert a['teacher_actions'].shape == (n, 2)
    else:
        assert not a['ownership'].any()
    return a


def _group(row, a, device):
    counts = np.asarray(a['native_counts'], dtype=np.float64)
    return dict(id=row['job_id'], x=torch.tensor(a['states'], device=device),
        y=torch.tensor(a['teacher_actions'], device=device, dtype=torch.float64),
        native_weight=torch.tensor(counts/counts.sum(), device=device, dtype=torch.float64),
        first=row['teacher_first'], prefix_decisions=row['teacher_first']+1)


def _matching_prefix(a, source, first):
    assert first < len(source['states'])
    for key in ['states', 'poses', 'hidden_before', 'history_poses',
                'history_goals', 'history_times']:
        assert np.array_equal(a[key][:first+1], source[key][:first+1]), key
    for key in ['actions', 'native_counts', 'next_states', 'end_flags']:
        assert np.array_equal(a[key][:first], source[key][:first]), key


def data(p, device='cuda'):
    initial, rejected = C._input_model(p)
    seeds, baselines = p['seed_occupancy_rows'], p['baseline_success_rows']
    assert len(seeds) == 6 and len(baselines) == 4
    assert {r['job_id'] for r in baselines} == OLD_IDS | WIDE_IDS
    students = seeds + baselines
    assert len({r['path'] for r in students}) == 10
    for row in students:
        _history(row, initial, safe=row in baselines)
        assert not row['teacher']
    originals = p['original_corrections']
    assert len(originals) == 2
    assert {r['job_id']: r['teacher_first'] for r in originals} == {
        'dev_task6_00': 26, 'dev_task6_21': 20}
    for row in originals:
        _history(row, initial, safe=True)
        assert row['teacher']
        if row['job_id'] == 'dev_task6_00':
            assert row['teacher_kind'] == 'safeT255reset'
            assert row['teacher_source_SHA256'] == p['expected_reset_teacher_digest']
            assert not row['teacher_hidden_burns_actual_history_from_h0']
        else:
            assert row['teacher_kind'] == 'T461_actual_history_carry'
            assert row['teacher_hidden_burns_actual_history_from_h0']
    # Only genuine original327 rows enter N.data's own-support path.
    batch = N.data(p, students, originals, 'occupancy', device=device)
    assert batch['correction_count'] == 2
    resets = p['new_reset_corrections']
    assert len(resets) == 2 and {r['job_id'] for r in resets} == OLD_IDS
    for row in resets:
        a = _history(row, rejected, safe=True)
        assert row['teacher'] and row['teacher_first'] == 20
        assert row['teacher_kind'] == 'safeT255reset'
        assert row['teacher_source_SHA256'] == p['expected_reset_teacher_digest']
        assert not row['teacher_hidden_burns_actual_history_from_h0']
        batch['corrections'].append(_group(row, a, device))
        batch['correction_provenance'].append(dict(id=row['job_id'],
            path=row['path'], sha256=row['sha256'], array_digest=row['array_digest'],
            source_policy_hash=row['policy_hash'], input_policy_hash=initial['policy_hash'],
            source_policy_matches_input=False, teacher_first=20,
            teacher_kind=row['teacher_kind'], teacher_source_SHA256=row['teacher_source_SHA256'],
            decisions=len(a['states']), physical_steps=row['physics'], old_support=False,
            heterogeneous_offpolicy_supervised_NOT_onpolicy=True,
            supervised_offpolicy_NOT_probability_ratio_data=True))
    batch['correction_count'] = len(batch['corrections'])
    assert batch['correction_count'] == 4
    for provenance, row in zip(batch['correction_provenance'], originals+resets):
        provenance.update(teacher_kind=row['teacher_kind'],
            teacher_source_SHA256=row['teacher_source_SHA256'], equal_trajectory_weight=1/4,
            full_actual_history_from_h0=True)
    wide_students = p['wide_counterexample_student_rows']
    wide_recoveries = p['wide_counterexample_recovery_rows']
    assert len(wide_students) == len(wide_recoveries) == 2
    assert {r['job_id'] for r in wide_students} == WIDE_IDS
    assert {r['job_id'] for r in wide_recoveries} == WIDE_IDS
    assert not ({r['path'] for r in wide_students+wide_recoveries+resets} &
                set(batch['current_own_support_paths']))
    sources = {r['job_id']: C._history(r, rejected) for r in wide_students}
    anchor_cpu = frozen(p, initial['path']).cpu().eval().requires_grad_(False)
    anchor = frozen(p, initial['path']).to(device).eval().requires_grad_(False)
    assert anchor_cpu.phash() == anchor.phash() == initial['policy_hash']
    groups, provenance = [], []
    for kind, rows in [('rejected_student', wide_students), ('validated_recovery', wide_recoveries)]:
        for row in rows:
            a = sources[row['job_id']] if kind == 'rejected_student' else C._history(row, rejected, True)
            if kind == 'validated_recovery':
                assert row['teacher_source_SHA256'] == initial['sha256']
                _matching_prefix(a, sources[row['job_id']], row['teacher_first'])
            x, target, check = C._target(anchor_cpu, anchor, a['states'], device)
            groups.append(dict(id=row['job_id'], kind=kind, x=x, target=target))
            provenance.append(dict(id=row['job_id'], kind=kind, path=row['path'],
                sha256=row['sha256'], array_digest=row['array_digest'],
                source_policy_hash=row['policy_hash'], target_policy_hash=initial['policy_hash'],
                target_model_sha256=initial['sha256'], decisions=len(x),
                physical_steps=row['physics'], equal_episode_weight=1/4,
                target='input327.sequence(full_actual_history_from_h0)_unclipped_location',
                offpolicy_REJECTED_diagnostic_model_NOT_newonpolicy=True,
                source_actions_are_NOT_training_targets=True, **check))
    assert len(groups) == 4
    batch.update(counterexample_groups=groups, counterexample_count=4,
        counterexample_provenance=provenance, counterexample_total_weight=1.,
        initial_counterexample_function=0., heterogeneous_teacher_corrections=True,
        offpolicy_REJECTED_diagnostic_model_NOT_newonpolicy=True)
    return batch


def bc_step(obj, batch, mode=None):
    mode = batch['mode'] if mode is None else mode
    assert mode == batch['mode'] == 'occupancy'
    q, opt = obj
    assert q._t463_mode == 'aggregate'
    C._parameters(q)
    opt.zero_grad(set_to_none=True)
    result = dict(mode=mode, student_prefix=0., teacher_suffix=0., duration_loss=0.,
        correction_objective=0., reference218=0., own10=0., counterexample_function=0.)
    count = len(batch['corrections'])
    assert count == batch['correction_count'] == 4
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
    assert len(groups) == batch['counterexample_count'] == 4
    assert batch['counterexample_total_weight'] == 1.
    for g in groups:
        loc = q.sequence(g['x'])
        assert loc.shape == g['target'].shape
        loss = (loc-g['target']).square().mean()/len(groups)
        loss.backward()
        result['counterexample_function'] += float(loss.detach())
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
        L._trainable(q), 1., error_if_nonfinite=True))
    C._parameters(q)
    opt.step()
    C._parameters(q)
    return result
