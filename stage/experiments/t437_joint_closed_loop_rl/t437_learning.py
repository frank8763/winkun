"""Finite Stage relay: fresh iid student PPO and verified recovery auxiliary.

The deployed Actor is unchanged. The first profiles train its existing4450
extra-MLP parameters; the finite representation fallback opens10722 existing
encoder/core/extra parameters. Static anchors are off-policy function
targets; real feedback controls are auxiliary labels, never PPO observations.
Full actual histories are evaluated before any teacher-suffix mask. Import is
inert: no simulator, model, optimizer, update, network or output-file operation.
"""
from pathlib import Path
import importlib.util
import gzip
import io

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'stage437_immutable432_learning',
    ROOT / 'stage/experiments/t432_offline_night/t432_learning.py')
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
E, torch, np = L.E, L.torch, L.np
EXECUTION, LABELS = L.EXECUTION, L.LABELS
TRAINABLE_COUNT, TRAINABLE_PREFIX = L.TRAINABLE_COUNT, L.TRAINABLE_PREFIX
load_complete_state = L.load_complete_state


def _stage(p):
    value = p.get('relay_stage', p.get('stage', p.get('phase_arm')))
    assert value in ['fresh_rl', 'iterative_recovery', 'representation_rl'], value
    return value


def configure(q, p):
    """Select a registered trainability profile without changing architecture."""
    q._relay_stage = _stage(p)
    params(q)
    return q


def trainable_prefixes(q):
    stage = getattr(q, '_relay_stage', 'fresh_rl')
    assert stage in ['fresh_rl', 'iterative_recovery', 'representation_rl']
    return ('encoder.', 'core.', 'extra.') if stage == 'representation_rl' else ('extra.',)


def trainable_count(q):
    return 10722 if getattr(q, '_relay_stage', 'fresh_rl') == 'representation_rl' else 4450


def params(q):
    q.requires_grad_(False)
    prefixes = trainable_prefixes(q)
    values = []
    for name, value in q.named_parameters():
        value.requires_grad_(name.startswith(prefixes))
        if value.requires_grad:
            values.append(value)
    assert sum(v.numel() for v in values) == trainable_count(q)
    assert all(v.requires_grad == name.startswith(prefixes) for name, v in q.named_parameters())
    return values


def fixedhash(q):
    prefixes = trainable_prefixes(q)
    return E.digest({name: value for name, value in q.state_dict().items()
                     if not name.startswith(prefixes)})


def make_optimizer(q, p):
    return torch.optim.Adam(params(q), lr=p['lr'], foreach=False, fused=False)


def export_model(q, opt, path):
    count, prefixes = trainable_count(q), trainable_prefixes(q)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad) == count
    state, hashes = E.L.state(q, opt), E.L.statehash(q, opt)
    buffer = io.BytesIO()
    torch.save(dict(state=state, hashes=hashes, policy_hash=q.phash(), memory_on=True,
                    architecture='unchanged_existing_single_Actor_registered_trainability',
                    relay_stage=getattr(q, '_relay_stage', 'fresh_rl'),
                    deployed_actor_count=1, trainable_parameters=count,
                    trainable_prefixes=prefixes, frozen_state_hash=fixedhash(q),
                    teacher_at_deployment=False), buffer)
    data = gzip.compress(buffer.getvalue(), mtime=0)
    assert len(data) <= 512 * 1024
    Path(path).write_bytes(data)
    return dict(path=str(path), sha256=E.sha(path), hashes=hashes, policy_hash=q.phash())


def _safe_target(q, entry, dynamic=False):
    arrays, first = entry['arrays'], entry['first']
    L._full(arrays)
    assert type(first) is int and 0 < first < len(arrays['states'])
    ownership = np.asarray(arrays['ownership'])
    assert not ownership[:first].any() and ownership[first:].all()
    assert np.array_equal(np.clip(arrays['locations'][:first], -1, 1).astype(np.float32),
                          arrays['actions'][:first])
    item = EXECUTION.recovery_group(q, arrays, first, entry['id'])
    item.update(dynamic_actual_student_recovery=bool(dynamic),
                source_policy_onpolicy_for_PPO=False)
    return item


def groups(q, p, anchors, targets, additional_targets=None):
    """Historical anchors keep original locations even after a new policy wins.

    ``additional_targets`` must be authenticated by the caller against its
    completed successful native recovery and exact replay before this call.
    Structural checks here independently reject truncated histories, bad
    ownership, collision, timeout and teacher controls outside their suffix.
    """
    stage = _stage(p)
    configure(q, p)
    probe = p.get('preflight_clone_probe') is True
    source_policy = p['expected_policy_hash']
    source_sha = p.get('anchor_source_model_sha256', p['student_start_sha256'])
    old_centres = {j['id'] for j in p['old8_jobs']}
    six_ids = {j['id'] for j in p['main_jobs']}
    extra_ids = {j['id'] for j in p['additional_old_jobs']}
    assert len(old_centres) == 8 and len(six_ids) == 54 and len(extra_ids) == 3
    assert extra_ids == {'dev_task6_08', 'dev_task6_09', 'dev_task6_10'}
    source_ids = {r['job_id'] for r in p['current_anchor_rows']}
    assert len(source_ids) == 59
    expected_ids = source_ids | extra_ids
    old, six, seen = [], [], set()
    for entry in anchors:
        arrays, row = LABELS._pair(entry)
        L._full(arrays)
        identity = row['job_id']
        assert identity not in seen and identity in expected_ids
        seen.add(identity)
        assert row['policy_hash'] == source_policy
        assert row['loaded_model_sha256'] == source_sha
        assert row['success'] and not row['collision'] and not row['timeout']
        assert not np.asarray(arrays['ownership']).any()
        assert np.array_equal(np.clip(arrays['locations'], -1, 1).astype(np.float32),
                              arrays['actions'])
        item = LABELS._label(q, arrays['states'], arrays['locations'], arrays['actions'])
        item.update(id=identity, actual_complete_history=True,
                    recorded_starting_student_location=True,
                    anchor_policy_hash=source_policy,
                    anchor_is_current_onpolicy=False,
                    historical_source_policy_verified=True)
        (six if identity in six_ids else old).append(item)
    if probe:
        assert len(anchors) == 2 and len(old) == len(six) == 1 and len(targets) == 1
    else:
        assert seen == expected_ids and len(old) == 11 and len(six) == 51
        assert len(targets) == 7

    history = E.N.old_data(q, E.N.science(p), lambda: None)
    assert len(history) == 211 and {v['task'] for v in history} == set(range(8))
    broad = []
    initial_policy = q.phash()
    for item in history:
        with torch.no_grad():
            location = q.sequence(item['x'], item['b']).detach().clone()
        label = LABELS._label(q, item['x'], location, location.clamp(-1, 1).float())
        label.update(id=item['id'], task=item['task'], self_distillation=True,
                     self_distillation_policy_hash=initial_policy,
                     historical_states_not_current_onpolicy=True)
        broad.append(label)

    corrections, seen_targets = [], set()
    expected_targets = set(p['recovery_target_ids'])
    assert len(expected_targets) == 7
    for target in targets:
        assert target['id'] in expected_targets and target['id'] not in seen_targets
        assert target['first'] in p['allowed_teacher_first']
        seen_targets.add(target['id'])
        corrections.append(_safe_target(q, target))
    if not probe:
        assert seen_targets == expected_targets
    dynamic = list(additional_targets or [])
    assert not dynamic or stage == 'iterative_recovery'
    assert len(dynamic) <= 2
    dynamic_ids = set()
    for target in dynamic:
        assert target['id'] not in dynamic_ids
        dynamic_ids.add(target['id'])
        corrections.append(_safe_target(q, target, dynamic=True))
    return dict(old=old, six=six, broad=broad, corrections=corrections,
                arm='b' if stage == 'fresh_rl' else 'c',
                relay_stage=stage, worst_case_preservation=stage != 'fresh_rl',
                trainable_parameters=trainable_count(q), trainable_prefixes=trainable_prefixes(q),
                old_weight=1., six_weight=1., broad_weight=1., correction_weight=1.,
                source_policy_hash=source_policy, stage_initial_policy_hash=initial_policy,
                source_anchor_ids=sorted(seen), recovery_target_ids=sorted(seen_targets),
                dynamic_recovery_ids=sorted(dynamic_ids),
                original_211_targets_are_starting_student_locations=True,
                actual_current_state_recovery_support=True,
                full_actual_history_before_teacher_suffix_mask=True)


make_groups = groups


def support_losses(q, g, p, k):
    assert g['relay_stage'] == getattr(q, '_relay_stage', 'fresh_rl')
    assert g['trainable_parameters'] == trainable_count(q)
    assert g['original_211_targets_are_starting_student_locations']
    assert g['old_weight'] == g['six_weight'] == g['broad_weight'] == g['correction_weight'] == 1.
    old_values = [E.old_kl(q, item, p) for item in g['old']]
    six_values = [E.old_kl(q, item, p) for item in g['six']]
    old_index = int(torch.argmax(torch.stack(old_values)).detach())
    six_index = int(torch.argmax(torch.stack(six_values)).detach())
    if g['worst_case_preservation']:
        old, six = old_values[old_index], six_values[six_index]
    else:
        old, six = sum(old_values) / len(old_values), sum(six_values) / len(six_values)
    choices = [item for item in g['broad'] if item['task'] == k % 8]
    assert choices
    broad = E.old_kl(q, choices[(k // 8) % len(choices)], p)
    correction = sum(EXECUTION.controlled_loss(q, item, p)
                     for item in g['corrections']) / len(g['corrections'])
    losses = [old, six, broad, correction]
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    details = dict(old_worst_id=g['old'][old_index]['id'],
                   six_worst_id=g['six'][six_index]['id'],
                   old_worst_loss=float(old_values[old_index].detach()),
                   six_worst_loss=float(six_values[six_index].detach()),
                   old_case_count=len(old_values), six_case_count=len(six_values),
                   reduction='max' if g['worst_case_preservation'] else 'mean')
    return losses, details


def _variables(q, opt):
    values = params(q)
    assert opt.defaults.get('foreach') is False and opt.defaults.get('fused') is False
    assert len(opt.param_groups) == 1
    assert [id(v) for v in opt.param_groups[0]['params']] == [id(v) for v in values]
    return values


def _frozen(q, expected):
    assert fixedhash(q) == expected
    assert all(v.grad is None for name, v in q.named_parameters()
               if not name.startswith(trainable_prefixes(q)))


def fit_bc(q, opt, g, p, guard, event, steps=40):
    assert steps == 40 or (steps == 1 and p.get('preflight_clone_probe') is True)
    assert g['relay_stage'] == 'iterative_recovery'
    assert g['dynamic_recovery_ids'] or p.get('preflight_clone_probe') is True
    variables, fixed = _variables(q, opt), fixedhash(q)
    trace, detail = [], []
    for k in range(steps):
        guard()
        opt.zero_grad(set_to_none=True)
        losses, choices = support_losses(q, g, p, k)
        reference = E.vector_gradient(losses[0] + losses[1] + losses[2], variables)
        correction = E.protect_vector(E.vector_gradient(losses[3], variables), reference)
        E.install_vector(variables, reference + correction)
        norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
        event('BEFORE', k)
        opt.step()
        event('AFTER', k)
        _frozen(q, fixed)
        trace.append([*[float(v.detach()) for v in losses], norm])
        detail.append(choices)
    return dict(trace=trace, per_case_preservation=detail, steps=steps, new_RL=0,
                trainable_parameters=trainable_count(q), fixed_state_hash=fixed,
                actual_executed_controls_only_no_teacher_latent=True,
                full_actual_history_before_teacher_suffix_mask=True,
                no_teacher_data_in_onpolicy_ratios=True)


def _student_batch(items):
    assert len(items) == 4
    for arrays in items:
        n = len(arrays['states'])
        assert n > 0 and np.asarray(arrays['states']).shape == (n, 120)
        assert np.asarray(arrays['next_states']).shape == (n, 120)
        assert np.array_equal(arrays['states'][1:], arrays['next_states'][:-1])
        assert not np.asarray(arrays['ownership']).any()
        assert np.asarray(arrays['stochastic']).shape == (n,)
        assert np.asarray(arrays['stochastic']).all()
        flags = np.asarray(arrays['end_flags'])
        assert flags.shape == (n, 3) and not flags[:-1].any()
        assert flags[-1].sum() == 1
        for name in ['states', 'next_states', 'raw_actions', 'locations', 'logp', 'rewards']:
            assert np.isfinite(arrays[name]).all(), name


def fit_ppo(q, opt, items, g, p, rnd, guard, event, steps=8):
    assert steps == 8 or (steps == 1 and p.get('preflight_clone_probe') is True)
    guard()
    _student_batch(items)
    # Existing strict step/sequence mean <1e-10 and iid log-density <1e-9
    # checks occur before the first optimizer event. They are not relaxed.
    batch, returns, errors = E.L.blocks(q, items, 0, p)
    assert len(batch) == 4
    variables, fixed = _variables(q, opt), fixedhash(q)
    trace, components, detail = [], [], []
    for k in range(steps):
        guard()
        opt.zero_grad(set_to_none=True)
        pg = 0.
        for x, b, raw, atom, oldlog, advantage in batch:
            ratio = (E.L.logp(q.sequence(x, b), raw, atom, p) - oldlog).exp()
            assert bool(torch.isfinite(ratio).all())
            coefficient = advantage / (4 * 400)
            loss = -torch.minimum(ratio * coefficient,
                                  ratio.clamp(1-p['ppo_clip'], 1+p['ppo_clip']) * coefficient).sum()
            assert bool(torch.isfinite(loss))
            loss.backward()
            pg += float(loss.detach())
        pg_vector = torch.cat([(torch.zeros_like(v) if v.grad is None else v.grad.detach().clone()).reshape(-1)
                               for v in variables])
        losses, choices = support_losses(q, g, p, rnd * 8 + k)
        reference = E.vector_gradient(losses[0] + losses[1] + losses[2], variables)
        correction = E.protect_vector(E.vector_gradient(losses[3], variables), reference)
        projected_pg = E.protect_vector(pg_vector, reference)
        E.install_vector(variables, reference + correction + projected_pg)
        norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
        event('BEFORE', k)
        opt.step()
        event('AFTER', k)
        _frozen(q, fixed)
        trace.append([pg, *[float(v.detach()) for v in losses], norm])
        components.append(dict(raw_PG_norm=float(pg_vector.norm()),
                               projected_PG_norm=float(projected_pg.norm()),
                               preservation_gradient_norm=float(reference.norm()),
                               projected_correction_gradient_norm=float(correction.norm())))
        detail.append(choices)
    with torch.no_grad():
        maximum = max(float((E.L.logp(q.sequence(x, b), raw, atom, p)-oldlog).abs().max())
                      for x, b, raw, atom, oldlog, _ in batch)
    return dict(trace=trace, per_case_preservation=detail,
                gradient_components=components, steps=steps, new_RL=steps,
                returns=returns.tolist(), density_errors=errors,
                post_max_abs_logratio=maximum,
                numeric_accept=bool(np.isfinite(maximum) and maximum <= p['max_logratio']),
                fixed_state_hash=fixed, trainable_parameters=trainable_count(q),
                full_actual_history_BPTT=True, teacher_data_only_auxiliary=True,
                onpolicy_training_ownership_zero=True,
                first_zero_no_forced_prefix=True,
                original_reward_episodic_peer_LOO=True,
                projection_is_local_soft_rule_not_closed_loop_guarantee=True)
