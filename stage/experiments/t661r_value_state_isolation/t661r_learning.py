"""Uniquely named baseline intervention; no new behavior data or Q."""
from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t660s_remaining_night_relay'))
import t660_learning as M


def load(obj, state):
    """Separate mutable Adam state for every model and saved-state owner."""
    for target, key in zip(obj, ['actor', 'value', 'actor_opt', 'value_opt']):
        target.load_state_dict(copy.deepcopy(state[key]))


def records(p):
    rows = M.read(p['source_rounds'])
    previous, current = rows[0], rows[1]
    assert not previous['research_state_kept'] and not current['research_state_kept']
    assert previous['start']['state_hash'] == current['start']['state_hash']
    assert M.read(p['source_audit'])['state'] == 'PASS'
    return previous, current


def initial_state(p, retain_value=True):
    previous, current = records(p)
    original = M.read_state(current['start'])
    donor = M.read_state(previous['trained'])
    result = copy.deepcopy(original)
    if retain_value:
        for key in ['value', 'value_opt']:
            result[key] = copy.deepcopy(donor[key])
    # The intervention cannot carry rejected Actor parameters or Adam.
    for key in ['actor', 'actor_opt']:
        assert M.digest(result[key]) == M.digest(original[key])
    return result


def groups(p, q, guard):
    previous, current = records(p)
    old_seeds = {tuple(r['seed']) for r in previous['training_rows']}
    new_seeds = {tuple(r['seed']) for r in current['training_rows']}
    assert not old_seeds & new_seeds and len(new_seeds) == 64
    result = []
    for job in p['training_jobs']:
        values = []
        for row in current['training_rows']:
            if row['job_id'] != job['id']:
                continue
            guard()
            assert row['policy_hash'] == q.phash() == p['initial']['policy_hash']
            assert row['loaded_model_sha256'] == p['initial']['sha256']
            assert M.sha(row['path']) == row['sha256']
            a = M.E.N.S.arrays(row['path'])
            assert all(M.digest(value) == row['array_hashes'][key] for key, value in a.items())
            assert not a['ownership'].any() and a['stochastic'].all()
            assert row['complete'] and not row['administrative_censor']
            values.append((a, row))
        assert len(values) == p['episodes_per_task'] == 2
        result.append((job['id'], values))
    assert len(result) == 32
    return result


def advantage_report(obj, data, p):
    items = M.fresh_items(obj, data, p)
    failed = [g for task in items for g in task if g['target_value'][0] == 0]
    good = [g for task in items for g in task if g['target_value'][0] == 1]
    return dict(
        episodes=sum(map(len, items)), failed_episodes=len(failed), successful_episodes=len(good),
        failed_with_nonzero_advantage=sum(bool(g['adv'].abs().max() > 0) for g in failed),
        failed_with_negative_advantage=sum(bool((g['adv'] < 0).any()) for g in failed),
        baseline_min=min(float(g['baseline'].min()) for task in items for g in task),
        baseline_max=max(float(g['baseline'].max()) for task in items for g in task),
        baseline_sha256=M.digest([g['baseline'] for task in items for g in task]),
        current_labels_not_used_to_fit_this_baseline=True,
        reused_data_is_on_policy_only_because_Actor_hash_is_identical=True,
    )
