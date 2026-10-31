"""Review completed T204 logs and saved weights; no simulator or optimizer."""
from pathlib import Path
import collections
import hashlib
import json
import os
import time

os.sched_setaffinity(0, {0})
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'stage/runs/t204_native_course'
OUT = ROOT / 'stage/runs/t204_progress_review'
DOCS = ROOT / 'stage/docs/results_20261031_native_course'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    p = SOURCE / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def safe(row):
    return row['complete'] and row['success'] and not row['collision']


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Review already exists; inspect it instead of duplicating'
    summary = json.loads((SOURCE / 'summary.json').read_text())
    plan = json.loads((SOURCE / 'protocol.json').read_text())
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped']
    rollouts, rounds, checks, novel = rows('rollouts.jsonl'), rows('rounds.jsonl'), rows('replay_checks.jsonl'), rows('new_capability.jsonl')
    last = max(r['round'] for r in rollouts if r.get('arm'))
    parent = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)['policy']
    arms = {}
    sources = [SOURCE / name for name in ['summary.json', 'protocol.json', 'rollouts.jsonl', 'rounds.jsonl', 'replay_checks.jsonl', 'new_capability.jsonl']]
    sources += [Path(plan['checkpoint']), Path(__file__)]
    for arm in plan['arms']:
        exported = Path(summary['exports'][arm]['path'])
        assert sha(exported) == summary['exports'][arm]['sha256']
        state = torch.load(exported, map_location='cpu', weights_only=False)['policy']
        accepted = SOURCE / 'candidates' / f'{arm}_r{last}_accepted.npz'
        with np.load(accepted) as z:
            keys = set(z.files)
            assert all(np.array_equal(state[k].numpy(), z[k]) for k in keys)
        assert all(torch.equal(state[k], v) for k, v in parent.items() if k not in keys)
        gates = [r for r in rollouts if r['id'].startswith(f'{arm}_r{last}_f1.0_gate_')]
        original = [r for r in rollouts if r['id'].startswith(f'{arm}_r{last}_post_primary_')]
        assert len(gates) == 7 and len(original) == 1
        for row in gates + original:
            path = Path(row['heads'])
            assert sha(path) == row['heads_sha256']
            with np.load(path) as z:
                assert all(np.array_equal(state[k].numpy(), z[k]) for k in keys)
            sources.append(path)
        training = [r for r in rollouts if r['role'] == 'training' and r['arm'] == arm]
        original_training = [r for r in training if r['task'] == 'primary_' + plan['primary_cases'][6]['id']]
        deterministic = [r for r in rollouts if r['mode'] == 'deterministic' and r['arm'] == arm and r['task'] == 'primary_' + plan['primary_cases'][6]['id']]
        interpolation = collections.Counter(str(m['chosen_interpolation']) for r in rounds for m in r['metrics'] if m['arm'] == arm)
        arms[arm] = {'saved_weights_match_last_accepted_heads': True,
                     'saved_backbone_and_other_parameters_match_parent': True,
                     'last_same_weight_old7_completed': len(gates), 'last_same_weight_old7_safe_GOAL': sum(safe(r) for r in gates),
                     'last_same_weight_LEFT_safe_GOAL': safe(original[0]), 'last_same_weight_LEFT_collision': original[0]['collision'],
                     'last_same_weight_LEFT_remaining_m': original[0]['remaining_m'],
                     'original_LEFT_training': len(original_training), 'original_LEFT_training_safe_GOAL': sum(safe(r) for r in original_training),
                     'original_LEFT_training_CONTACT': sum(r['collision'] for r in original_training),
                     'original_LEFT_deterministic_cases': len(deterministic), 'original_LEFT_deterministic_safe_GOAL': sum(safe(r) for r in deterministic),
                     'completed_round_interpolation_counts': dict(interpolation),
                     'evidence_scope': 'Previously recorded final-round trajectories use exactly the saved full model weights. Separate final/reload battery not executed; no new rollout.'}
        sources += [exported, accepted]
    pending = [r for r in checks if not r['verified']]
    assert len(pending) == 1
    mismatches = [r for r in checks if r['pose_error'] != 0 or r['action_error'] >= 2e-6 or r['logp_error'] >= 1e-5]
    result = {'state': 'COMPLETE', 'decision': 'STOP_UNCHANGED_T204_TRAINING',
              'physics_steps': 0, 'optimizer_updates': 0,
              'training_already_stopped': True, 'T204_stop_reason': summary['stop_reason'],
              'full_round_boundaries': len(rounds), 'training_rounds_collected': len({r['round'] for r in rollouts if r['role'] == 'training'}),
              'training_episodes': summary['stats']['training_episodes'], 'actor_updates': summary['stats']['actor_updates'],
              'arms': arms, 'original_LEFT_training_safe_GOAL_total': sum(a['original_LEFT_training_safe_GOAL'] for a in arms.values()),
              'original_LEFT_training_total': sum(a['original_LEFT_training'] for a in arms.values()),
              'separate_final_old8_battery_completed': len(summary['final_evaluation']),
              'robustness_completed': summary['robustness_completed'],
              'replay_verified': sum(r['verified'] for r in checks), 'replay_wall_censored': len(pending), 'replay_mismatches': mismatches,
              'unique_new_capability_tasks': len({r['task'] for r in novel}),
              'new_capability_details': [{'task': r['task'], 'round': r['round'], 'arm': r['arm'],
                                         'initial_goal_distance_m': r['parent']['initial_distance_m'],
                                         'parent_nearest_m': r['parent']['nearest_m'], 'candidate_nearest_m': r['candidate']['nearest_m'],
                                         'parent_miss_above_radius_m': r['parent']['nearest_m'] - .2,
                                         'final_candidate_retention_not_tested': True} for r in novel],
              'source_reset_gap_m': 4.645468288486047, 'best_ever_pool_reset_gap_m': rounds[-1]['best_distance_to_original_reset_m'],
              'pool_is_union_of_historical_success_not_revalidated_final_pool': True,
              'training_fraction_of_total_physics': summary['stats']['training_physics'] / summary['total_physics'],
              'next_step': 'Keep official T159 and T194. Before any new training, target the unbridged LEFT failure frontier and measure sequence-level exploration and learning signal there; use a separate bounded protocol and reserve final evaluation time. Do not automatically resume T204 or start long-distance.',
              'input_hashes': {str(p): sha(p) for p in sorted(set(sources))}}
    assert time.monotonic() - started < 60
    OUT.mkdir()
    text = json.dumps(result, indent=2) + '\n'
    (OUT / 'review.json').write_text(text)
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 1024**2
    assert sum(p.stat().st_size for p in (ROOT / 'stage/runs').rglob('*') if p.is_file()) < 1024**3
    (DOCS / 'progress_review.json').write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
