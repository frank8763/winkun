"""T432 bounded offline-night learner: representation depth and task retention.

Both registered arms train only the EXISTING 136->32->2 extra MLP (4450
parameters). No deployed module/capacity, recurrent state, base policy or
runtime teacher is added. B keeps mean-preservation; C uses the worst complete
old/six preservation loss and three extra authentic old-task trajectories.

Import is inert. Native/source/audit authentication belongs to the registered
caller; only full actual-history safe-GOAL arrays may form learning groups.
"""
from pathlib import Path
import gzip
import importlib.util
import io

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'stage432_immutable431_data_contract',
    ROOT / 'stage/experiments/t431_actual_recovery/t431_learning.py')
PREVIOUS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PREVIOUS)

E, torch, np = PREVIOUS.E, PREVIOUS.torch, PREVIOUS.np
EXECUTION = PREVIOUS.EXECUTION
LABELS = PREVIOUS.PREVIOUS
load_complete_state = PREVIOUS.load_complete_state
TRAINABLE_PREFIX = 'extra.'
TRAINABLE_COUNT = 4450


def params(q):
    """Train existing extra.0 and extra.2 only; GRU/base/adapter stay fixed."""
    q.requires_grad_(False)
    named = dict(q.named_parameters())
    assert named['extra.0.weight'].shape == (32, 136)
    assert named['extra.0.bias'].shape == (32,)
    assert named['extra.2.weight'].shape == (2, 32)
    assert named['extra.2.bias'].shape == (2,)
    for name, value in named.items():
        value.requires_grad_(name.startswith(TRAINABLE_PREFIX))
    result = [v for name, v in q.named_parameters() if name.startswith(TRAINABLE_PREFIX)]
    assert sum(v.numel() for v in result) == TRAINABLE_COUNT
    assert len(result) == 4
    assert all(v.requires_grad == name.startswith(TRAINABLE_PREFIX)
               for name, v in q.named_parameters())
    return result


def fixedhash(q):
    return E.digest({name: value for name, value in q.state_dict().items()
                     if not name.startswith(TRAINABLE_PREFIX)})


def _full(arrays):
    PREVIOUS._full_history(arrays)
    assert not np.asarray(arrays['end_flags'])[:, 0].any()


def groups(q, p, anchors, targets):
    """Full source-policy anchors plus seven real current-state recoveries."""
    params(q)
    arm = p['arm']
    assert arm in ['b', 'c']
    policy = q.phash()
    assert policy == p['expected_policy_hash']
    probe = p.get('preflight_clone_probe') is True
    old_centres = {j['id'] for j in p['old8_jobs']}
    six_ids = {j['id'] for j in p['main_jobs']}
    extra_ids = {j['id'] for j in p['additional_old_jobs']}
    assert len(old_centres) == 8 and len(six_ids) == 54 and len(extra_ids) == 3
    assert old_centres.isdisjoint(six_ids) and extra_ids.isdisjoint(old_centres | six_ids)
    assert extra_ids == {'dev_task6_08', 'dev_task6_09', 'dev_task6_10'}
    source_rows = p['current_anchor_rows']
    source_ids = {r['job_id'] for r in source_rows}
    assert len(source_rows) == len(source_ids) == 59
    assert len(source_ids & old_centres) == 8 and len(source_ids & six_ids) == 51
    expected_ids = source_ids | (extra_ids if arm == 'c' else set())
    target_ids = set(p['recovery_target_ids'])
    assert len(target_ids) == 7
    assert set(p['allowed_teacher_first']) == {101, 76, 116, 78, 73, 80, 110}
    old, six, seen = [], [], set()
    for entry in anchors:
        arrays, row = LABELS._pair(entry)
        _full(arrays)
        identity = row['job_id']
        assert identity not in seen and identity in expected_ids
        seen.add(identity)
        assert row['policy_hash'] == policy and row['loaded_model_sha256'] == p['student_start_sha256']
        assert row['success'] and not row['collision'] and not row['timeout']
        assert not np.asarray(arrays['ownership']).any()
        assert np.array_equal(np.clip(arrays['locations'], -1, 1).astype(np.float32), arrays['actions'])
        item = LABELS._label(q, arrays['states'], arrays['locations'], arrays['actions'])
        item.update(id=identity, actual_complete_history=True,
                    recorded_starting_student_location=True,
                    current_source_policy_verified=True,
                    additional_current_old_support=identity in extra_ids)
        (six if identity in six_ids else old).append(item)
    if probe:
        assert len(anchors) == 2 and len(old) == len(six) == 1 and len(targets) == 1
    else:
        assert seen == expected_ids
        assert len(old) == (8 if arm == 'b' else 11) and len(six) == 51
        assert len(targets) == 7

    history = E.N.old_data(q, E.N.science(p), lambda: None)
    assert len(history) == 211 and {item['task'] for item in history} == set(range(8))
    broad = []
    for item in history:
        with torch.no_grad():
            location = q.sequence(item['x'], item['b']).detach().clone()
        target = LABELS._label(q, item['x'], location, location.clamp(-1, 1).float())
        target.update(id=item['id'], task=item['task'], self_distillation=True,
                      self_distillation_policy_hash=policy,
                      historical_states_not_current_onpolicy=True)
        broad.append(target)

    corrections, seen_targets = [], set()
    for target in targets:
        identity, first = target['id'], target['first']
        assert identity in target_ids and identity not in seen_targets
        seen_targets.add(identity)
        arrays = target['arrays']
        _full(arrays)
        assert type(first) is int and first in p['allowed_teacher_first']
        assert 0 < first < len(arrays['states'])
        assert not arrays['ownership'][:first].any() and arrays['ownership'][first:].all()
        assert np.array_equal(np.clip(arrays['locations'][:first], -1, 1).astype(np.float32),
                              arrays['actions'][:first])
        corrections.append(EXECUTION.recovery_group(q, arrays, first, identity))
    if not probe:
        assert seen_targets == target_ids
    return dict(old=old, six=six, broad=broad, corrections=corrections,
                arm=arm, worst_case_preservation=arm == 'c',
                trainable_parameters=TRAINABLE_COUNT, trainable_prefix=TRAINABLE_PREFIX,
                old_weight=1., six_weight=1., broad_weight=1., correction_weight=1.,
                source_policy_hash=policy, source_anchor_ids=sorted(seen),
                recovery_target_ids=sorted(seen_targets),
                original_211_targets_are_starting_student_locations=True,
                actual_current_state_recovery_support=True,
                full_actual_history_before_teacher_suffix_mask=True)


make_groups = groups


def support_losses(q, g, p, k):
    assert g['arm'] == p['arm'] and g['trainable_parameters'] == TRAINABLE_COUNT
    assert g['original_211_targets_are_starting_student_locations']
    assert g['old_weight'] == g['six_weight'] == g['broad_weight'] == g['correction_weight'] == 1.
    old_values = [E.old_kl(q, item, p) for item in g['old']]
    six_values = [E.old_kl(q, item, p) for item in g['six']]
    # Deterministic list order breaks exact ties. Do not relabel max-trajectory
    # preservation as per-step bounds or a closed-loop safety guarantee.
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
    detail = dict(old_worst_id=g['old'][old_index]['id'],
                  six_worst_id=g['six'][six_index]['id'],
                  old_worst_loss=float(old_values[old_index].detach()),
                  six_worst_loss=float(six_values[six_index].detach()),
                  old_case_count=len(old_values), six_case_count=len(six_values),
                  reduction='max' if g['worst_case_preservation'] else 'mean')
    return losses, detail


def make_optimizer(q, p):
    return torch.optim.Adam(params(q), lr=p['lr'], foreach=False, fused=False)


def fit(q, opt, g, p, guard, event, steps=120):
    assert steps == 120 or (steps == 1 and p.get('preflight_clone_probe') is True)
    assert opt.defaults.get('foreach') is False and opt.defaults.get('fused') is False
    variables = params(q)
    assert len(opt.param_groups) == 1
    assert [id(v) for v in opt.param_groups[0]['params']] == [id(v) for v in variables]
    fixed = fixedhash(q)
    trace, per_case = [], []
    for k in range(steps):
        guard()
        opt.zero_grad(set_to_none=True)
        losses, details = support_losses(q, g, p, k)
        values = [float(loss.detach()) for loss in losses]
        reference = E.vector_gradient(losses[0] + losses[1] + losses[2], variables)
        correction = E.protect_vector(E.vector_gradient(losses[3], variables), reference)
        E.install_vector(variables, reference + correction)
        norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
        event('BEFORE', k)
        opt.step()
        event('AFTER', k)
        trace.append([*values, norm])
        per_case.append(details)
        assert all(v.grad is None for name, v in q.named_parameters()
                   if not name.startswith(TRAINABLE_PREFIX))
        if k % 16 == 0:
            assert fixedhash(q) == fixed
    assert fixedhash(q) == fixed
    return dict(trace=trace, per_case_preservation=per_case, steps=steps,
                new_RL=0, arm=p['arm'], trainable_parameters=TRAINABLE_COUNT,
                trainable_prefix=TRAINABLE_PREFIX, fixed_state_hash=fixed,
                frozen_encoder_core_base_head_adapter_and_all_other_tensors=True,
                unchanged_deployed_architecture=True,
                current_old_anchor_count=len(g['old']), six_anchor_count=len(g['six']),
                recovery_count=len(g['corrections']),
                original_211_targets_are_starting_student_locations=True,
                full_actual_history_before_teacher_suffix_mask=True,
                actual_executed_controls_only_no_teacher_latent=True,
                no_teacher_data_in_onpolicy_ratios=True,
                worst_trajectory_loss_not_a_safety_certificate=True,
                projection_is_local_soft_rule_not_closed_loop_guarantee=True)


def export_model(q, opt, path):
    assert sum(v.numel() for v in q.parameters() if v.requires_grad) == TRAINABLE_COUNT
    state, hashes = E.L.state(q, opt), E.L.statehash(q, opt)
    buffer = io.BytesIO()
    torch.save(dict(state=state, hashes=hashes, policy_hash=q.phash(), memory_on=True,
        architecture='unchanged_existing_T431_single_Actor_extra_MLP_trainable_only',
        deployed_actor_count=1, trainable_parameters=TRAINABLE_COUNT,
        trainable_prefix=TRAINABLE_PREFIX, frozen_state_hash=fixedhash(q),
        teacher_at_deployment=False), buffer)
    data = gzip.compress(buffer.getvalue(), mtime=0)
    assert len(data) <= 512 * 1024
    Path(path).write_bytes(data)
    return dict(path=str(path), sha256=E.sha(path), hashes=hashes, policy_hash=q.phash())
