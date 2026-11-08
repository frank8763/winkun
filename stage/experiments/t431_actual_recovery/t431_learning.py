"""Stage T431: actual-state recovery data for the unchanged 66-parameter fit.

Only the group builder changes. The T427 output-head selection, loss,
optimizer, fixed120-step proposal, shadow contract and model codec are reused
without modifying their historical source. Recovery controls are actual
off-policy executions; their complete observation prefix is retained for
causal carry, but only the owned suffix supplies action labels.

Import is inert: no Actor, simulator, optimizer, fit or filesystem write is
performed. Source SHA/audit authentication remains the registered caller's
responsibility; arrays must already be decoded by E.N.S.arrays.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    'stage431_immutable427_learning',
    ROOT / 'stage/experiments/t427_fixed_representation/t427_learning.py')
PREVIOUS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PREVIOUS)

E, torch, np = PREVIOUS.E, PREVIOUS.torch, PREVIOUS.np
EXECUTION = PREVIOUS.EXECUTION
TRAINABLE_NAMES = PREVIOUS.TRAINABLE_NAMES
params = PREVIOUS.params
fixedhash = PREVIOUS.fixedhash
make_optimizer = PREVIOUS.make_optimizer
support_losses = PREVIOUS.support_losses
fit = PREVIOUS.fit
export_model = PREVIOUS.export_model
load_complete_state = PREVIOUS.load_complete_state


def _full_history(arrays):
    """Require decoded, complete real safe-GOAL observation histories."""
    PREVIOUS._safe_arrays(arrays)
    n = len(arrays['states'])
    assert np.asarray(arrays['next_states']).shape == (n, 120)
    assert np.array_equal(arrays['states'][1:], arrays['next_states'][:-1])
    assert np.asarray(arrays['hidden_before']).shape == (n, 16)
    assert np.asarray(arrays['history_poses']).shape == (n + 1, 4, 3)
    assert np.asarray(arrays['history_goals']).shape == (n + 1, 4, 2)
    assert np.asarray(arrays['history_times']).shape == (n + 1, 4)
    assert np.asarray(arrays['locations']).shape == (n, 2)
    assert np.asarray(arrays['ownership']).shape == (n,)
    assert np.asarray(arrays['stochastic']).shape == (n,)
    for name in ['next_states', 'hidden_before', 'history_poses',
                 'history_goals', 'history_times', 'locations']:
        assert np.isfinite(arrays[name]).all(), name
    assert not np.asarray(arrays['stochastic']).any()


def groups(q, p, anchors, targets):
    """Keep current51six/8old histories and label six or seven real recoveries.

    Production IDs are locked by anchor_jobs and recovery_target_ids. An
    explicitly marked preflight uses exactly one current-policy old anchor,
    one current-policy six anchor and one authenticated real recovery. It
    does not relax source-policy, history, terminal or ownership checks.
    """
    params(q)
    policy = q.phash()
    assert policy == p['expected_policy_hash']
    probe = p.get('preflight_clone_probe') is True
    old_ids = {job['id'] for job in p['old8_jobs']}
    six_ids = {job['id'] for job in p['main_jobs']}
    assert len(old_ids) == 8 and len(six_ids) == 54 and not old_ids & six_ids
    expected_anchors = [job['id'] for job in p['anchor_jobs']]
    assert len(expected_anchors) == len(set(expected_anchors)) == 59
    assert len(set(expected_anchors) & old_ids) == 8
    assert len(set(expected_anchors) & six_ids) == 51
    assert set(expected_anchors) <= old_ids | six_ids
    expected_targets = list(p['recovery_target_ids'])
    assert len(expected_targets) in [6, 7]
    assert len(expected_targets) == len(set(expected_targets))
    allowed_first = set(p['allowed_teacher_first'])
    assert allowed_first == {101, 76, 116, 78, 73, 80, 110}
    assert anchors and targets

    old, six, seen = [], [], set()
    for entry in anchors:
        arrays, row = PREVIOUS._pair(entry)
        _full_history(arrays)
        identity = row['job_id']
        assert identity not in seen and identity in expected_anchors
        seen.add(identity)
        assert row['policy_hash'] == policy
        assert row['loaded_model_sha256'] == p['student_start_sha256']
        assert row['success'] and not row['collision'] and not row['timeout']
        assert not np.asarray(arrays['ownership']).any()
        assert np.array_equal(np.clip(arrays['locations'], -1, 1).astype(np.float32),
                              arrays['actions'])
        item = PREVIOUS._label(q, arrays['states'], arrays['locations'], arrays['actions'])
        item.update(id=identity, actual_complete_history=True,
                    recorded_starting_student_location=True,
                    historical_preflight_fixture=probe,
                    current_source_policy_verified=True)
        (old if identity in old_ids else six).append(item)
    if probe:
        assert len(anchors) == 2 and len(old) == len(six) == 1
        assert len(targets) == 1
    else:
        assert seen == set(expected_anchors)
        assert len(old) == 8 and len(six) == 51
        assert len(targets) == len(expected_targets)

    history = E.N.old_data(q, E.N.science(p), lambda: None)
    assert len(history) == 211 and {item['task'] for item in history} == set(range(8))
    broad = []
    for item in history:
        with torch.no_grad():
            location = q.sequence(item['x'], item['b']).detach().clone()
        target = PREVIOUS._label(q, item['x'], location, location.clamp(-1, 1).float())
        target.update(id=item['id'], task=item['task'], self_distillation=True,
                      self_distillation_policy_hash=policy,
                      historical_states_not_current_onpolicy=True)
        broad.append(target)

    corrections, ids = [], set()
    for target in targets:
        identity = target['id']
        assert identity not in ids and identity in expected_targets
        ids.add(identity)
        arrays, first = target['arrays'], target['first']
        _full_history(arrays)
        assert type(first) is int and first in allowed_first
        assert 0 < first < len(arrays['states'])
        ownership = np.asarray(arrays['ownership'])
        assert not ownership[:first].any() and np.all(ownership[first:] == 1)
        assert np.array_equal(np.clip(arrays['locations'][:first], -1, 1).astype(np.float32),
                              arrays['actions'][:first])
        # recovery_group evaluates the entire actual history; its unchanged
        # controlled_loss masks only after causal sequence evaluation.
        corrections.append(EXECUTION.recovery_group(q, arrays, first, identity))
    if not probe:
        assert ids == set(expected_targets)
    return dict(old=old, six=six, broad=broad, corrections=corrections,
                old_weight=1., six_weight=1., broad_weight=1., correction_weight=1.,
                fixed_representation=True, trainable_parameters=66,
                original_211_targets_are_starting_student_locations=True,
                historical_preflight_fixture_only=probe,
                actual_current_state_recovery_support=True,
                source_policy_hash=policy,
                recovery_target_ids=sorted(ids))


make_groups = groups
