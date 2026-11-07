"""T422 recovery-control supervision and separately sampled recurrent PPO.

The feedback controller has no neural latent action or hidden state.  Its
supervised targets are the bounded controls actually executed during a full,
verified safe-GOAL recovery.  The deterministic control surrogate below is
not a teacher likelihood, an on-policy density, or a safety certificate.

All recurrent predictions use the complete actual observation history; a
suffix mask selects labels without resetting the student's hidden state.
This module performs no work at import time and never launches simulation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t418_evening_repair_chain'))
import repair_core as E

torch, np = E.torch, E.np


def recovery_group(q, arrays, first, source_id=None):
    """Make an execution-only target; require a complete real safe GOAL."""
    flags = np.asarray(arrays['end_flags'])
    actions = np.asarray(arrays['actions'])
    states = np.asarray(arrays['states'])
    first = int(first)
    assert flags.shape == (len(states), 3)
    assert flags[-1, 1] and not flags[:, 0].any() and not flags[-1, 2]
    assert not flags[:-1].any()
    assert 0 <= first < len(states) and states.shape[1:] == (120,)
    assert actions.shape == (len(states), 2)
    assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
    if 'ownership' in arrays:
        assert not np.asarray(arrays['ownership'])[:first].any()
        assert np.asarray(arrays['ownership'])[first:].all()
    x = torch.as_tensor(states).detach().clone()
    with torch.no_grad():
        b = q.base_actions(x).detach().clone()
    return dict(x=x, b=b,
                actions=torch.as_tensor(actions, dtype=torch.float64).detach().clone(),
                first=first, source_id=source_id, executed_teacher=True,
                actual_complete_history=True, safe_goal_verified=True,
                no_teacher_latent=True, not_onpolicy=True)


def build_groups(q, initial_arrays, p, recoveries, old_correction_entries):
    """Keep all initial successes, broad old support, and verified labels.

    ``recoveries`` is a list of dictionaries containing ``arrays`` and
    ``first`` (optionally ``source_id``).  Their native/source SHA and exact
    replay verification remain the caller's responsibility.  ``old`` entries
    retain T418's original checked neural-teacher likelihood contract.
    """
    g = E.build_groups(q, initial_arrays, p)
    assert recoveries, 'No BC without a newly verified feedback recovery'
    corrections = []
    for entry in old_correction_entries:
        item = E.checked_query(q, entry, p)
        item.update(executed_teacher=False, original_checked_query=True)
        corrections.append(item)
    for entry in recoveries:
        assert isinstance(entry, dict) and 'arrays' in entry and 'first' in entry
        corrections.append(recovery_group(q, entry['arrays'], entry['first'],
                                          entry.get('source_id')))
    assert any(item.get('executed_teacher', False) for item in corrections)
    g['corrections'] = corrections
    return g


def executed_residual(mean, actions):
    """Squared interior error and one-sided loss at exact bounded atoms.

    Saturated execution +1 only asks for mean >= +1, and execution -1 only
    asks for mean <= -1.  This avoids inventing an unclipped controller
    location or pulling an already saturated mean back to the boundary.
    It is a deterministic action-target surrogate, not a probability loss.
    """
    actions = torch.as_tensor(actions, dtype=mean.dtype, device=mean.device)
    assert actions.shape == mean.shape and mean.shape[-1] == 2
    assert bool(torch.isfinite(actions).all()) and float(actions.abs().max()) <= 1
    interior = (mean - actions).square()
    return torch.where(actions == 1, torch.relu(1 - mean).square(),
                       torch.where(actions == -1, torch.relu(mean + 1).square(),
                                   interior))


def controlled_loss(q, item, p):
    assert item.get('executed_teacher') and item.get('safe_goal_verified')
    assert item.get('no_teacher_latent') and item.get('not_onpolicy')
    prediction = q.sequence(item['x'], item['b'])
    assert prediction.shape == item['actions'].shape
    first = int(item['first'])
    assert 0 <= first < len(prediction)
    # The sequence is evaluated from observation zero.  Masking after that
    # preserves causal hidden carry and BPTT through the actual prefix.
    loss = executed_residual(prediction[first:], item['actions'][first:]).mean()
    assert bool(torch.isfinite(loss))
    return loss


def target_loss(q, item, p):
    return controlled_loss(q, item, p) if item.get('executed_teacher') else E.nll(q, item, p)


def support_losses(q, g, p, k):
    assert g['anchors'] and len(g['broad']) == 211 and len(g['left']) == 5
    old = sum(E.old_kl(q, item, p) for item in g['anchors']) / len(g['anchors'])
    old = old * p.get('old_weight', 1.)
    options = [item for item in g['broad'] if item['task'] == k % 8]
    assert options
    broad = E.old_kl(q, options[(k // 8) % len(options)], p) * p.get('broad_old_weight', 1.)
    left = E.nll(q, g['left'][k % 5], p) * p.get('left_weight', .25)
    right = E.nll(q, g['right'], p) * p.get('right_weight', .25)
    corrections = g.get('corrections', [])
    assert corrections and any(item.get('executed_teacher') for item in corrections)
    current = sum(target_loss(q, item, p) for item in corrections) / len(corrections)
    current = current * p.get('query_weight', 1.)
    return [old, broad, left, right, current]


def backward_auxiliary(losses, variables, pg=None):
    """Old-gradient priority is a local gradient rule, not a safety proof."""
    old = E.vector_gradient(losses[0] + losses[1], variables)
    projected = [E.protect_vector(E.vector_gradient(loss, variables), old)
                 for loss in losses[2:] if loss.requires_grad]
    total = old + sum(projected, torch.zeros_like(old))
    if pg is not None:
        total = total + E.protect_vector(pg, old)
    E.install_vector(variables, total)


def _check_parameters(q, variables):
    assert q.contract['phase'] in ['protect', 'rl']
    assert sum(v.numel() for v in variables) == 10722
    assert all(v.requires_grad == name.startswith(E.trained_prefixes(q))
               for name, v in q.named_parameters())


def fit_bc(q, opt, g, p, guard, event, steps=200):
    count = int(steps)
    assert 0 < count <= 200
    assert any(item.get('executed_teacher') for item in g.get('corrections', []))
    variables = E.params(q)
    _check_parameters(q, variables)
    fixed = E.original_hash(q)
    trace = []
    for k in range(count):
        guard()
        opt.zero_grad(set_to_none=True)
        losses = support_losses(q, g, p, k)
        values = [float(loss.detach()) for loss in losses]
        backward_auxiliary(losses, variables)
        norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
        event('BEFORE', k)
        opt.step()
        event('AFTER', k)
        trace.append([*values, norm])
        assert all(v.grad is None for name, v in q.named_parameters()
                   if not name.startswith(E.trained_prefixes(q)))
        if k % 32 == 0:
            assert E.original_hash(q) == fixed
    assert E.original_hash(q) == fixed
    return dict(trace=trace, steps=count, new_RL=0,
                trainable_parameters=sum(v.numel() for v in variables),
                original_frozen_hash=fixed,
                own_initial_success_histories_each_step=True,
                full_actual_history_before_suffix_label_mask=True,
                feedback_recovery_groups=sum(bool(item.get('executed_teacher')) for item in g['corrections']),
                original_query_groups=sum(not bool(item.get('executed_teacher')) for item in g['corrections']),
                query_groups=len(g['corrections']),
                feedback_loss='squared_actual_execution_target; exact_boundary_one_sided_hinge; no_teacher_latent_or_density',
                old_gradient_priority=True)


def fit_ppo(q, opt, items, g, p, rnd, guard, event):
    # No externally controlled teacher trajectory is admitted to PPO ratios.
    assert len(items) == 4
    for item in items:
        assert not np.asarray(item['ownership']).any()
        assert np.asarray(item['stochastic']).all()
    epochs = int(p.get('epochs', 8))
    assert 0 < epochs <= 8
    batch, returns, errors = E.L.blocks(q, items, 0, p)
    variables = E.params(q)
    _check_parameters(q, variables)
    fixed = E.original_hash(q)
    trace, components = [], []
    for k in range(epochs):
        guard()
        opt.zero_grad(set_to_none=True)
        pg = 0.
        for x, b, raw, atom, oldlog, adv in batch:
            ratio = (E.L.logp(q.sequence(x, b), raw, atom, p) - oldlog).exp()
            coefficient = adv / (4 * 400)
            loss = -torch.minimum(ratio * coefficient,
                                  ratio.clamp(1 - p['ppo_clip'], 1 + p['ppo_clip']) * coefficient).sum()
            assert bool(torch.isfinite(loss))
            loss.backward()
            pg += float(loss.detach())
        before = [v.grad.detach().clone() for v in variables]
        pgflat = torch.cat([value.reshape(-1) for value in before])
        losses = support_losses(q, g, p, int(rnd) * epochs + k)
        backward_auxiliary(losses, variables, pgflat)
        aux = [v.grad.detach() - value for v, value in zip(variables, before)]
        pg_norm = float(sum(value.square().sum() for value in before).sqrt())
        aux_norm = float(sum(value.square().sum() for value in aux).sqrt())
        dot = float(sum((a * b).sum() for a, b in zip(before, aux)))
        norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
        event('BEFORE', k)
        opt.step()
        event('AFTER', k)
        trace.append([pg, *[float(loss.detach()) for loss in losses], norm])
        components.append(dict(pg_norm=pg_norm, aux_norm=aux_norm,
                               pg_aux_cosine=None if not pg_norm * aux_norm else dot / (pg_norm * aux_norm)))
        assert all(v.grad is None for name, v in q.named_parameters()
                   if not name.startswith(E.trained_prefixes(q)))
        assert E.original_hash(q) == fixed
    with torch.no_grad():
        maximum = max(float((E.L.logp(q.sequence(x, b), raw, atom, p) - oldlog).abs().max())
                      for x, b, raw, atom, oldlog, _ in batch)
    return dict(trace=trace, steps=epochs, gradient_components=components,
                returns=returns.tolist(), density_errors=errors,
                post_max_abs_logratio=maximum,
                numeric_accept=bool(np.isfinite(maximum) and maximum <= p['max_logratio']),
                original_frozen_hash=fixed,
                trainable_parameters=sum(v.numel() for v in variables),
                teacher_data_only_auxiliary=True,
                own_initial_success_histories_each_step=True,
                full_actual_history_before_suffix_label_mask=True,
                no_teacher_actions_in_PPO_ratios=True,
                old_gradient_priority=True)
