"""Read-only aggregation of completed Stage pilots; never steps a simulator."""
from pathlib import Path
from collections import Counter
import hashlib
import json

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'stage/runs'
OUT = REPO / 'stage/docs/results_20261030'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    checks = []
    for name in ('t170_vector_td3', 't171_reverse_curriculum'):
        root = RUNS / name
        assert json.loads((root / 'pipeline_status.json').read_text())['state'] == 'COMPLETE'
        status = json.loads((root / 'status.json').read_text())
        assert sha(status['checkpoint']) == status['checkpoint_sha256']
        protocol = json.loads((REPO / 'stage/experiments' / name / 'protocol.json').read_text())
        for p, digest in protocol['input_hashes'].items():
            assert sha(p) == digest, p
        result = json.loads((root / 'frozen_eval/summary.json').read_text())
        assert all(result['actor_unchanged'].values())
        paired = {}
        for row in result['cases']:
            paired.setdefault(row['id'], set()).add((row['task_signature'], row['scene_signature']))
            assert not (row['safe_success'] and row['collision'])
            assert row['step_simulation_s'] == .002
            assert row['steps'] <= row['step_cap']
        assert len(paired) == 12 and all(len(v) == 1 for v in paired.values())
        checks.append({'run': name, 'hashes_and_frozen_pairing': 'PASS',
                       'checkpoint_sha256': status['checkpoint_sha256']})

    root = RUNS / 't171_reverse_curriculum'
    episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
    phases = []
    for phase in range(4):
        rows = [r for r in episodes if r['task']['phase'] == phase]
        canonical = [r for r in rows if r['task']['canonical5m']]
        phases.append({'phase': phase, 'completed_episodes': len(rows),
                       'success': sum(r['success'] for r in rows),
                       'collisions': sum(r['collision'] for r in rows),
                       'ends': dict(Counter(r['end'] for r in rows)),
                       'canonical_5m_episodes': len(canonical),
                       'canonical_5m_success': sum(r['success'] for r in canonical),
                       'minimum_start_distance_m': min(r['task']['distance_m'] for r in rows),
                       'maximum_start_distance_m': max(r['task']['distance_m'] for r in rows)})
    failures = [{k: row[k] for k in ('id', 'group', 'end', 'remaining_m', 'nearest_m',
                                    'simulation_duration_s')}
                for row in json.loads((root / 'frozen_eval/summary.json').read_text())['cases']
                if row['model'] == 'T171_final' and not row['safe_success']]
    result = {'verification': checks, 't171_training_by_assigned_reset_phase': phases,
              't171_final_frozen_failures': failures,
              'interpretation_limits': [
                  'Noisy online successes are not frozen-policy evaluation.',
                  'Episodes spanning a phase transition retain their reset-phase label.',
                  'At the budget boundary active episodes are censored, not scored as failures.',
                  'Phase3 has only2 completed episodes; no stable long-range success estimate.',
                  'The four same-map3m cases were consumed inT170 and are development cases inT171.',
                  'All runs use one geometric layout and one training seed each.'],
              'retained_actor': 'T159', 'original_5m_safe_success': '0/2',
              'automatic_promotion': False}
    (OUT / 't170_t171_verification_and_frontier.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps({'state': 'PASS', 'training_by_phase': phases, 'retained_actor': 'T159'}))


if __name__ == '__main__':
    main()
