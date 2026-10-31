"""Read completed evidence only; reconcile counters and report actual coverage."""
from pathlib import Path
import collections
import hashlib
import json
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'stage/runs/t204_native_course'
DOCS = ROOT / 'stage/docs/results_20261031_native_course'


def lines(name):
    path = OUT / name
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


def safe(row):
    return row['complete'] and row['success'] and not row['collision']


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped']
    plan = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in plan['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    rows = lines('rollouts.jsonl')
    replay = lines('replay_checks.jsonl')
    rounds = lines('rounds.jsonl')
    novel = lines('new_capability.jsonl')
    compact = {r['id']: r for r in lines('compaction.jsonl')}
    physics = collections.Counter()
    for row in rows:
        physics['training' if row['role'] == 'training' else 'diagnostic'] += row['physics_steps']
    physics['replay'] = sum(r['physics_steps'] for r in replay)
    for category in ['training', 'diagnostic', 'replay']:
        assert physics[category] == summary['stats'][category + '_physics']
    assert sum(physics.values()) == summary['total_physics']
    arms = {}
    for arm in plan['arms']:
        training = [r for r in rows if r['arm'] == arm and r['role'] == 'training']
        original = [r for r in training if r['task'] == 'primary_' + plan['primary_cases'][6]['id']]
        final = [r for r in summary['final_evaluation'] if r['id'].startswith(arm + '_final_')]
        actions, means = [], []
        for row in training:
            path = compact[row['id']]['compact_path'] if row['id'] in compact else row['path']
            if path:
                with np.load(path) as z:
                    actions.append(z['actions']); means.append(z['means'])
        coverage = {}
        if actions:
            action = np.concatenate(actions); mean = np.concatenate(means)
            clip = mean[:, 0] >= .999999
            coverage = {'decisions': len(action), 'native_mean_at_or_above_upper_bound': int(clip.sum()),
                        'clip_throttle_below_0_9_fraction': float((action[clip, 0] < .9).mean()) if clip.any() else None,
                        'throttle_quantiles': np.quantile(action[:, 0], [0, .1, .5, .9, 1]).tolist()}
        arms[arm] = {'training_episodes': len(training), 'training_safe_GOAL': sum(safe(r) for r in training),
                     'training_CONTACT': sum(r['collision'] for r in training),
                     'training_TIMEOUT': sum(r['timeout'] for r in training),
                     'original_LEFT_training_episodes': len(original),
                     'original_LEFT_training_safe_GOAL': sum(safe(r) for r in original),
                     'final_old8_safe_GOAL': sum(safe(r) for r in final),
                     'final_old8': [{'case': r['case']['id'], 'safe_GOAL': safe(r), 'collision': r['collision'],
                                     'nearest_m': r['nearest_m'], 'remaining_m': r['remaining_m']} for r in final],
                     'novel_success_tasks': sorted({r['task'] for r in novel if r['arm'] == arm}),
                     'behavior_coverage': coverage}
    result = {'state': 'COMPLETE', 'stop_reason': summary['stop_reason'], 'physics': dict(physics),
              'stats': summary['stats'], 'rounds_completed': len(rounds), 'arms': arms,
              'unique_new_capability_tasks': len({r['task'] for r in novel}),
              'source_reset_gap_m': 4.645468288486047,
              'final_pool_gap_m': rounds[-1]['best_distance_to_original_reset_m'] if rounds else None,
              'original_LEFT_passed': summary['original_LEFT_passed'],
              'stable_5m_gate_passed': summary['stable_5m_gate_passed'],
              'robustness_completed': summary['robustness_completed'],
              'robustness_safe_GOAL': summary['robustness_safe_GOAL'],
              'all_replay_checks_verified': all(r['verified'] for r in replay),
              'replay_checks': len(replay), 'elapsed_s': summary['elapsed_s'],
              'limitations': 'One paired seed, shared adaptive curriculum, fixed map and development tasks. Head-only adaptation; deployed network is end-to-end. No independent generalization or innovation claim.'}
    folder = OUT / 'analysis'; folder.mkdir(exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    if rounds:
        x = [r['round'] + 1 for r in rounds]
        ax[0].plot([0] + x, [result['source_reset_gap_m']] + [r['best_distance_to_original_reset_m'] for r in rounds], 'o-', ms=3)
        for arm in plan['arms']:
            y = [next(m['postupdate_safe_GOAL'] / 2 for m in r['metrics'] if m['arm'] == arm) for r in rounds]
            ax[1].plot(x, y, 'o-', ms=3, label=arm)
    ax[0].set(xlabel='Completed round', ylabel='Successful reset distance to original start (m)', title='Shared reset pool; lower is closer')
    ax[1].set(xlabel='Completed round', ylabel='Post-update task safe GOAL fraction', ylim=(-.05, 1.05), title='Two common course tasks per round')
    ax[1].legend()
    for axis in ax:
        axis.grid(alpha=.25)
    fig.savefig(folder / 'course_progress.png', dpi=180)
    plt.close(fig)
    own = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert own < plan['max_bytes']
    assert sum(p.stat().st_size for p in (ROOT / 'stage/runs').rglob('*') if p.is_file()) < 1024**3
    shutil.copyfile(folder / 'metrics.json', DOCS / 'metrics.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
