"""Independent zero-navigation reconciliation after T259 has stopped."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'stage/runs/t259_closed_loop_recovery'
DOC = ROOT/'stage/docs/long_distance_20261101'
sys.path.insert(0, str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L, S


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    p = OUT/name
    return list(map(json.loads, p.read_text().splitlines())) if p.exists() else []


def safe(row):
    return row['complete'] and row['success'] and not row['collision']


def main():
    receipt = DOC/'T259_AUDIT.json'
    assert not receipt.exists(), 'Audit already recorded; do not overwrite'
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['all_workers_stopped']
    workers = json.loads((OUT/'workers.json').read_text())
    for pid in [workers['parent'], *workers['workers']]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f'Process still present or PID reused: {pid}')
    names = ['baseline.jsonl', 'training.jsonl', 'replays.jsonl',
             'candidate_guards.jsonl', 'round_centers.jsonl', 'old_regression.jsonl',
             'development.jsonl', 'heldout.jsonl', 'verification.jsonl', 'next_distance.jsonl']
    data = {name: rows(name) for name in names}
    geometry = rows('geometry.jsonl')
    accounted_physics = sum(r['physics_steps'] for rs in data.values() for r in rs)
    accounted_commands = len(geometry) + sum(map(len, data.values()))
    assert accounted_physics == summary['physics_steps']
    assert accounted_commands == summary['commands']
    updates = rows('updates.jsonl')
    assert len(updates) == summary['rounds'] and all(r['shadow_exact'] for r in updates)
    assert sum(u['accepted'] for r in updates for u in r['updates']) == summary['accepted_RL_updates']
    assert summary['accepted_RL_updates'] == summary['shadow_accepted_updates']
    train, replay = data['training.jsonl'], data['replays.jsonl']
    assert len(train) == summary['unique_training_episodes']
    assert len(replay) == summary['exact_replays'] == len(train)
    for a, b in zip(train, replay):
        assert a['id'] == b['id'] and a['array_hashes'] == b['array_hashes']
        assert a['integration_sha256'] == b['integration_sha256']
    for name, nkey, skey in [('old_regression.jsonl','old_regression_n','old_regression_safe'),
                              ('development.jsonl','new_development_n','new_development_safe'),
                              ('heldout.jsonl','heldout_n','heldout_safe')]:
        rr = data[name]
        assert len(rr) == summary[nkey] and sum(map(safe, rr)) == summary[skey]
        assert len({r['source_reset_id'] for r in rr}) == len(rr)
    files = {}
    for rs in data.values():
        for row in rs:
            if row.get('path'):
                path = row['path']
                assert sha(path) == row['sha256']
                files[path] = row['sha256']
    plan = json.loads((OUT/'protocol.json').read_text())
    assert (OUT/'protocol.json').read_bytes() == (ROOT/'stage/experiments/t259_closed_loop_recovery/protocol.json').read_bytes()
    for name, expected in plan['input_hashes'].items():
        assert sha(name) == expected, name
    export = summary['export']
    assert sha(export['path']) == export['sha256']
    payload = L.read(export['path'])
    assert S.digest(payload['policy']) == export['policy_hash']
    assert S.digest(payload['optimizer']) == export['optimizer_hash']
    events = rows('proposal_events.jsonl')
    accepts = [next(a for a in e['attempts'] if a.get('accept'))
               for e in events if e['phase']=='live' and e.get('accepted')]
    prior = list(map(json.loads, (ROOT/'stage/runs/t257_distance_policy_gradient/training.jsonl').read_text().splitlines()))
    signatures = {(json.dumps(r['array_hashes'],sort_keys=True),r['integration_sha256']) for r in prior}
    overlap = sum((json.dumps(r['array_hashes'],sort_keys=True),r['integration_sha256']) in signatures for r in train)
    assert overlap == summary['exact_historical_training_overlap']
    result = dict(state='POSTRUN_EVIDENCE_RECONCILED', physics_steps_added=0,
        optimizer_updates_added=0, all_processes_absent=True,
        source_commit=workers['source_commit'], original_summary_sha256=sha(OUT/'summary.json'),
        protocol_sha256=sha(OUT/'protocol.json'), physics_from_raw_rows=accounted_physics,
        commands_from_raw_rows=accounted_commands,
        unaccounted_inflight_physics_upper_bound=summary['inflight_physics_upper_bound'],
        training_records=len(train), exact_replays=len(replay),
        exact_historical_overlap=overlap, retained_updates=summary['accepted_RL_updates'],
        accepted_live_events_outside_T257_bounds=sum(not a['would_pass_T257_function_bounds'] for a in accepts),
        all_recorded_episode_payload_hashes_verified=len(files), final_model=export,
        old_safe=summary['old_regression_safe'], old_n=summary['old_regression_n'],
        six_m_safe=summary['new_development_safe'], six_m_n=summary['new_development_n'],
        six_m_by_side={side:dict(n=sum(r['job']['case']['side']==side for r in data['development.jsonl']),
            safe=sum(safe(r) and r['job']['case']['side']==side for r in data['development.jsonl']))
            for side in ['LEFT','RIGHT']}, heldout_executed=summary['heldout_n'],
        reload_verification_executed=summary['reload_exact_verification_n'],
        six_m_gate=summary['new_distance_gate'], seven_m_test_executed=len(summary['next_distance']),
        independent_training_seed_count=1, source_baseline_frozen=True,
        audit_epoch=time.time())
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
