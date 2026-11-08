"""POSTHOC adapter for a FUTURE separately registered recovery-learning pilot.

Import is inert. Reading validated existing arrays does not launch a learner,
construct an Actor, or change any model. No successful new label is invented.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / 'stage/docs/feedback_20261108/RECOVERY_SUPPORT.json'
CASES = ROOT / 'stage/runs/t427_fixed_representation/learn/cases.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def unpack(path):
    """Existing lossless state-chain layout; never modify the source NPZ."""
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if 'last_next_state' in arrays:
        assert 'next_states' not in arrays
        last = arrays.pop('last_next_state')
        assert last.shape == (1, 120) and last.dtype == arrays['states'].dtype
        arrays['next_states'] = np.concatenate([arrays['states'][1:], last])
    return arrays


def materialize(expected_index_sha256, expected_policy_hash):
    """Validate source identity, success, owner mask, and full causal history.

    Future callers must lock both arguments in their protocol. They must
    rebuild targets if their initialization policy differs from this source.
    Teacher controls are execution targets, never student PPO samples.
    """
    assert sha(INDEX) == expected_index_sha256
    index = read(INDEX)
    assert index['state'] == 'VERIFIED_RECOVERY_LABEL_INDEX_NO_TRAINING'
    items = index['successes']
    assert len(items) == 6 and len({i['job_id'] for i in items}) == 6
    result = []
    for item in items:
        assert item['policy_hash'] == expected_policy_hash
        assert item['on_policy_for_PPO'] is False
        path = Path(item['path'])
        assert path.resolve().is_relative_to((ROOT / 'stage/runs').resolve())
        assert sha(path) == item['sha256']
        assert sha(item['source_audit_path']) == item['source_audit_sha256']
        assert read(item['source_audit_path'])['state'] == 'PASS'
        arrays = unpack(path)
        n = len(arrays['actions'])
        first = item['first_feedback_or_teacher_hold']
        assert type(first) is int and 0 <= first < n
        assert arrays['states'].shape == (n, 120)
        assert arrays['hidden_before'].shape == (n, 16)
        assert arrays['history_poses'].shape == (n + 1, 4, 3)
        assert arrays['history_goals'].shape == (n + 1, 4, 2)
        assert arrays['history_times'].shape == (n + 1, 4)
        assert not arrays['ownership'][:first].any() and arrays['ownership'][first:].all()
        assert not arrays['stochastic'].any()
        assert np.array_equal(arrays['end_flags'][-1], [0, 1, 0])
        assert not arrays['end_flags'][:-1].any()
        assert np.array_equal(arrays['states'][1:], arrays['next_states'][:-1])
        assert np.array_equal(np.clip(arrays['locations'][:first], -1, 1).astype(np.float32),
                              arrays['actions'][:first])
        assert np.isfinite(arrays['states']).all() and np.isfinite(arrays['actions']).all()
        assert np.abs(arrays['actions']).max() <= 1
        result.append(dict(id=item['job_id'], arrays=arrays, first=first,
                           path=str(path), sha256=item['sha256'],
                           actual_complete_history=True, not_onpolicy=True,
                           successful_current_state_recovery=True))
    return result


def current_success_anchors(expected_cases_sha256, expected_policy_hash):
    """59 full current-policy successes, not all 271 successful old tasks."""
    assert sha(CASES) == expected_cases_sha256
    rows = [json.loads(line) for line in CASES.read_text().splitlines()]
    rows = [r for r in rows if r['full'] and r['success'] and not r['collision']]
    assert len(rows) == 59 and len({r['job_id'] for r in rows}) == 59
    result = []
    for row in rows:
        assert row['policy_hash'] == expected_policy_hash and sha(row['path']) == row['sha256']
        arrays = unpack(row['path'])
        assert not arrays['ownership'].any() and not arrays['stochastic'].any()
        assert np.array_equal(arrays['end_flags'][-1], [0, 1, 0])
        assert np.array_equal(np.clip(arrays['locations'], -1, 1).astype(np.float32), arrays['actions'])
        result.append(dict(arrays=arrays, row=row))
    return result


if __name__ == '__main__':
    index = read(INDEX)
    policy = index['successes'][0]['policy_hash']
    recoveries = materialize(sha(INDEX), policy)
    anchors = current_success_anchors(sha(CASES), policy)
    assert sum(len(r['arrays']['actions']) - r['first'] for r in recoveries) == 1039
    print(json.dumps(dict(state='EXISTING_DATA_ADAPTER_PASS_NOT_A_TRAINING_LAUNCH',
        recoveries=len(recoveries), anchors=len(anchors), teacher_action_labels=1039,
        index_sha256=sha(INDEX), cases_sha256=sha(CASES), source_policy_hash=policy,
        source_sha256=sha(__file__), native_steps=0, Actor_forwards=0,
        Actor_updates=0, optimizer_calls=0, no_parameters_or_gradients_constructed=True), indent=2))
