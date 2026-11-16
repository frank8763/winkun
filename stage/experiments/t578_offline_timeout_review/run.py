"""Read saved Stage evidence only. Never imports a simulator, policy or optimizer.

coverage: a red-capable saved-trace counterexample to small-set retention.
analyze: one immutable offline review; missing poses stay unknown.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'stage/runs/t578_offline_timeout_review'
DOC = ROOT / 'stage/docs/day_20261116/t578'
NIGHT = ROOT / 'stage/runs/t577_offline_recovery_night'
OLD = ROOT / 'stage/runs/t550_offline_obstacle_relay/final407'
SOURCES = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def rows(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def arrays(row):
    path = Path(row['path'])
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256'], path
    SOURCES[str(path.relative_to(ROOT))] = row['sha256']
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    assert int(a['native_counts'].astype(np.int64).sum()) == row['physics']
    return a


def safe(row):
    return bool(row['complete'] and row['success'] and not row['collision']
                and row['field_detail']['field_qualified_success'])


def tally(rr):
    return dict(n=len(rr), success=sum(safe(r) for r in rr),
                contact=sum(bool(r['collision']) for r in rr),
                timeout=sum(bool(r['timeout']) for r in rr),
                censored=sum(bool(r.get('administrative_censor')) for r in rr))


def quantile(values):
    return None if not values else dict(zip(['q10', 'median', 'q90'],
        map(float, np.quantile(values, [.1, .5, .9]))))


def coverage():
    check = read(NIGHT / 'rl_03_round/summary.json')['result']['check']
    rr = rows(NIGHT / 'final407/cases.jsonl')
    assert check['accepted'] and not check['lost_prior']
    assert not check['reference23_missing'] and len(rr) == 407
    bad = [r['job_id'] for r in rr if not safe(r)]
    print(json.dumps(dict(small_gate='PASS', full407=tally(rr),
        implication_small_gate_guarantees_full407=not bad,
        counterexample=bad[0] if bad else None)))
    return 2 if bad else 0


def trajectory(row, a):
    dt = a['native_counts'].astype(float) * .002
    # This step duration is checked against the actually saved physical clock.
    np.testing.assert_allclose(np.diff(a['history_times'][:, -1]), dt,
                               atol=2e-8, rtol=0)
    poses = np.vstack([a['poses'], a['next_poses'][-1]])
    np.testing.assert_allclose(poses[1:], a['next_poses'], atol=1e-12, rtol=0)
    d = np.linalg.norm(poses[:, :2] - a['task_goal_xy'], axis=1)
    assert abs(float(d[-1]) - row['remaining_m']) < 1e-8
    displacement = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    speed = displacement / dt
    radial = (d[:-1] - d[1:]) / dt
    t = np.r_[0., np.cumsum(dt)]
    # Descriptive thresholds fixed before this review; not new success gates.
    return dict(job=row['job_id'], role=row['role'], policy_hash=row['policy_hash'],
        teacher=bool(row['teacher']), success=safe(row), collision=row['collision'],
        timeout=row['timeout'], seconds=float(t[-1]), start_distance_m=float(d[0]),
        remaining_m=float(d[-1]), min_decision_distance_m=float(d.min()),
        closest_time_s=float(t[d.argmin()]), path_at_decision_endpoints_m=float(displacement.sum()),
        mean_endpoint_speed_m_s=float(displacement.sum() / dt.sum()),
        fraction_time_endpoint_speed_below_0_04=float(dt[speed < .04].sum() / dt.sum()),
        moving_away_seconds=float(dt[radial < -.001].sum()),
        near_half_m_then_exit=bool(d.min() <= .5 and d[-1] > .5),
        command_drive_mean=float(np.average(a['actions'][:, 0], weights=dt)),
        source_path=row['path'], source_sha256=row['sha256'],
        not_a_final_model_autonomous_suite_measurement=True)


def analyze():
    start = time.monotonic()
    assert not RUN.exists(), 'Never overwrite or restart a completed review'
    assert set(os.sched_getaffinity(0)) <= {14, 15}
    used = sum(p.stat().st_size for p in (ROOT / 'stage/runs').rglob('*') if p.is_file())
    assert used + 2**20 < 2**30 and shutil.disk_usage(ROOT).free >= 40 * 2**30
    config = read(NIGHT / 'protocol.json')
    current = rows(NIGHT / 'final407/cases.jsonl')
    previous = rows(OLD / 'cases.jsonl')
    assert len(current) == len(previous) == 407
    old_by = {r['job_id']: r for r in previous}
    groups = [('old275', 0, 275), ('six54', 275, 329),
              ('eight54', 329, 383), ('field24', 383, 407)]
    transition, suites, full_array_schemas = [], {}, collections.Counter()
    for name, lo, hi in groups:
        pairs = collections.Counter()
        metrics = collections.defaultdict(list)
        rr = current[lo:hi]
        for row in rr:
            old = old_by[row['job_id']]
            assert row['request']['job'] == old['request']['job']
            for key in ['initial_native_integration', 'scene_sha256', 'task_goal_xy']:
                assert row['array_hashes'][key] == old['array_hashes'][key]
            before, after = safe(old), safe(row)
            pairs[f'{int(before)}->{int(after)}'] += 1
            a = arrays(row)
            full_array_schemas[','.join(sorted(a))] += 1
            outcome = 'success' if after else ('timeout' if row['timeout'] else 'contact')
            weights = a['native_counts'].astype(float)
            metrics[outcome].append(dict(id=row['job_id'], remaining_m=row['remaining_m'],
                seconds=float(weights.sum() * .002),
                drive_command_time_mean=float(np.average(a['actions'][:, 0], weights=weights)),
                steer_saturated_time_fraction=float(np.average(abs(a['actions'][:, 1]) >= .95,
                                                                weights=weights))))
            if before != after:
                transition.append(dict(id=row['job_id'], suite=name, old_safe=before,
                    new_safe=after, new_timeout=row['timeout'], remaining_m=row['remaining_m']))
        suites[name] = dict(current=tally(rr), reference=tally(previous[lo:hi]),
            paired_transition=dict(pairs), outcomes={k: dict(n=len(v),
                remaining_m=quantile([x['remaining_m'] for x in v]),
                command_drive_time_mean=quantile([x['drive_command_time_mean'] for x in v]),
                steer_saturated_fraction=quantile([x['steer_saturated_time_fraction'] for x in v]),
                terminal_within_half_m=sum(x['remaining_m'] <= .5 for x in v))
                for k, v in metrics.items()})
    full = []
    for f in sorted(NIGHT.glob('*/cases.jsonl')):
        if not (f.parent.name.startswith('rl_') or f.parent.name == 'mean_00_recovery'):
            continue
        for row in rows(f):
            if row['full']:
                full.append(dict(child=f.parent.name, **trajectory(row, arrays(row))))
    rounds = []
    for f in sorted(NIGHT.glob('rl_*_round/summary.json')):
        r = read(f)['result']
        rounds.append(dict(child=f.parent.name, BC=len(r['BC']), RL=r['new_RL'],
            accepted=r['accepted'], check=r['check']))
    stress = {0: collections.Counter(), 1: collections.Counter()}
    paired = collections.Counter()
    for f in sorted(NIGHT.glob('stress_*/stress_results.jsonl')):
        rr = rows(f)
        assert len(rr) == 32
        for a, b in zip(rr[::2], rr[1::2]):
            assert a['model'] == 0 and b['model'] == 1 and a['pair'] == b['pair']
            paired[f"{int(a['success'])}->{int(b['success'])}"] += 1
            for x in [a, b]:
                c = stress[x['model']]; c['n'] += 1
                for key in ['success', 'collision', 'timeout', 'censored', 'boundary_crossed']:
                    c[key] += bool(x[key])
    # Fixed representatives only; these are NOT registered or authorized for simulation.
    filters = [lambda r: r['job_id'].startswith('dev_task3_') and r['timeout'],
               lambda r: r['job_id'].startswith('wide117_80_dev6_LEFT_') and r['timeout'],
               lambda r: r['job_id'].startswith('wide117_80_dev6_RIGHT_') and r['timeout'],
               lambda r: r['job_id'].startswith('wide117_8m_dev6_LEFT_') and r['timeout'],
               lambda r: r['job_id'].startswith('wide117_8m_dev6_RIGHT_') and r['timeout'],
               lambda r: r['job_id'].startswith('t502_') and r['timeout']]
    chosen = []
    for predicate in filters:
        pool = sorted([r for r in current if predicate(r)], key=lambda r:(r['remaining_m'],r['job_id']))
        assert pool
        chosen.append(pool[(len(pool)-1)//2])
    proposed_physics = 6 * sum(r['request']['job']['case']['step_cap'] for r in chosen)
    assert proposed_physics <= 2400000
    assert time.monotonic() - start < 600
    protocol = dict(id='T578_OFFLINE_TIMEOUT_REVIEW', authorization='User: offline analysis only; no simulation',
        new_native=0, new_optimizer=0, new_policy_inference=0, output_max_bytes=2**20,
        wall_max_seconds=600, CPU=[14], GPU=False,
        command_semantics='Drive is a normalized command, not measured speed; negative is not proof of reverse',
        thresholds=dict(near_m=.5, low_endpoint_speed_m_s=.04, away_radial_m_s=-.001),
        inputs=SOURCES)
    report = dict(state='COMPLETE', new_native=0, new_optimizer=0, new_policy_inference=0,
        all407_initial_native_scene_goal_jobs_match_reference=True, suites=suites,
        changed_outcomes=transition, saved_final_array_schemas=dict(full_array_schemas),
        rounds=rounds, full_trace_descriptions=full,
        frozen_stress=dict(models={str(k):dict(v) for k,v in stress.items()}, paired_transition=dict(paired)),
        next_diagnostic_draft=dict(executed=False, authorized=False,
            jobs=[r['request']['job'] for r in chosen],
            selection='Within six fixed task families, lower median terminal distance among timeouts, tie by ID',
            models=['T550 retained reference','T576 nonlinear initial','T577 rl03 retained research'],
            unique=18, exact=18, max_physics=proposed_physics, max_commands=40,
            full_trajectory_required=True, budget_not_granted=True),
        limitations=['All407 compact files contain only actions/native_counts/end_flags; no full pose inference.',
            'Training and mixed teacher traces are not autonomous final-policy performance.',
            'No initial T576 full407, so cannot attribute all reference-relative losses to T577 updates.',
            'Drive associations and sampled endpoint speeds do not establish collision or timeout causality.',
            'Perturbation summaries have digests, not independent native exact or saved full history.'])
    output = {'protocol.json':protocol, 'summary.json':report}
    blobs = {k:json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n' for k,v in output.items()}
    assert sum(map(len,blobs.values())) < 2**20
    for f,h in SOURCES.items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h, f
    RUN.mkdir()
    for name,b in blobs.items():
        with (RUN/name).open('xb') as f:f.write(b)
    print(json.dumps(dict(state='COMPLETE', new_native=0, new_optimizer=0,
        paired_losses=sum(x['old_safe'] and not x['new_safe'] for x in transition),
        paired_gains=sum(x['new_safe'] and not x['old_safe'] for x in transition),
        full_traces=len(full), output_bytes=sum(map(len,blobs.values())),
        draft_physics=proposed_physics),ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['coverage']:sys.exit(coverage())
    assert sys.argv[1:] == ['analyze']
    analyze()
