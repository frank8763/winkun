"""One locked prefix of existing T259 episodes; zero simulation or learning."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t261_distance_credit_audit'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    start = time.monotonic()
    plan = json.loads((HERE / 'protocol.json').read_text())
    assert not (OUT / 'summary.json').exists(), 'No repeat or enlarged prefix'
    for name, expected in plan['input_hashes'].items():
        assert sha(Path(name).read_bytes()) == expected, name
    lines = Path(plan['training_rows']).read_bytes().splitlines(keepends=True)[:plan['episodes']]
    assert len(lines) == plan['episodes'] and sha(b''.join(lines)) == plan['prefix_sha256']
    rows = [json.loads(line) for line in lines]
    geometry = {r['id']: r for r in map(json.loads, Path(plan['geometry']).read_text().splitlines())}
    gamma = math.exp(-.002 / 60)
    records = []
    returns = []
    for row in rows:
        assert time.monotonic() - start < plan['wall_seconds']
        assert row['complete']
        path = Path(row['path'])
        assert sha(path.read_bytes()) == row['sha256']
        with np.load(path) as z:
            reward = z['rewards'].astype(np.float64)
            counts = z['native_counts'].astype(np.int64)
            flags = z['end_flags'][-1]
            actions = z['actions'].astype(np.float64)
            means = z['means'].astype(np.float64)
        n = int(counts.sum())
        assert n == row['physics_steps'] and len(reward) == len(counts)
        acc = 0.
        ret = np.empty(len(reward), dtype=np.float64)
        for k in range(len(ret) - 1, -1, -1):
            acc = float(reward[k]) + gamma ** int(counts[k]) * acc
            ret[k] = acc
        times = np.r_[0, np.cumsum(counts)[:-1]]
        direct = float(np.dot(gamma ** times, reward))
        g = geometry[row['source_reset_id']]
        initial = float(np.linalg.norm(np.array(g['goal_xy']) - np.array(g['pose'][:2])))
        event = float(flags[1]) - float(flags[0]) - .5 * float(flags[2])
        time_cost = -.000002 * (1 - gamma ** n) / (1 - gamma)
        terminal = gamma ** (n - 1) * event
        analytic = .03 * initial + time_cost + terminal
        assert abs(acc - direct) < 1e-11 and abs(acc - analytic) < 1e-9
        returns.append(ret)
        records.append(dict(id=row['id'], reset=row['source_reset_id'],
            round=int(row['id'].split('_')[0][1:]), side=row['job']['case']['side'],
            distance_m=row['job']['case']['distance_m'], safe=row['success'] and not row['collision'],
            collision=row['collision'], timeout=row['timeout'],
            duration_s=.002*n, cap_s=.002*row['job']['case']['step_cap'],
            remaining_m=row['remaining_m'], initial_distance_m=initial, return_at_start=acc,
            time_cost=time_cost, discounted_terminal_event=terminal,
            analytic_return_error=abs(acc-analytic), direct_return_error=abs(acc-direct),
            command_minus_clipped_saved_mean=np.average(actions-np.clip(means,-.999999,.999999),
                                                        axis=0,weights=counts).tolist(),
            historical_overlap=row.get('exact_historical_overlap', False)))
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record['round'], record['reset']), []).append(index)
    group_reports = []
    for (rnd, reset), indices in groups.items():
        assert len(indices) == 4
        for j in indices:
            baseline = np.mean([returns[k][0] for k in indices if k != j])
            advantage = returns[j] - baseline
            records[j].update(leave_one_out_baseline=float(baseline),
                advantage_at_start=float(advantage[0]),
                positive_advantage_fraction=float(np.mean(advantage > 0)),
                min_advantage=float(advantage.min()), max_advantage=float(advantage.max()))
        successful = [j for j in indices if records[j]['safe']]
        failed = [j for j in indices if not records[j]['safe']]
        group_reports.append(dict(round=rnd, reset=reset, safe=len(successful),
            success_returns_above_all_failures=(min(returns[j][0] for j in successful) >
                max(returns[j][0] for j in failed)) if successful and failed else None,
            returns=[float(returns[j][0]) for j in indices]))
    mixed = [g for g in group_reports if g['success_returns_above_all_failures'] is not None]
    result = dict(state='COMPLETE', episodes=len(records), physics_steps=0, optimizer_updates=0,
        source_prefix_sha256=plan['prefix_sha256'], episodes_detail=records, groups=group_reports,
        mixed_groups=len(mixed), mixed_groups_success_ranked_highest=sum(
            g['success_returns_above_all_failures'] for g in mixed),
        max_analytic_return_error=max(r['analytic_return_error'] for r in records),
        historical_overlap=sum(r['historical_overlap'] for r in records),
        interpretation='Read-only fixed-prefix reward/credit audit. No policy gradients, parameter '
            'updates, Actor reconstruction or simulated actions. Return ordering does not prove '
            'low gradient variance, causal algorithm improvement or deterministic success. '
            'Command-minus-clipped-mean is a saved-state descriptive proxy, not Gaussian AR '
            'innovation or a closed-loop counterfactual. Full T259 gates remain authoritative.',
        elapsed_s=time.monotonic()-start)
    data = (json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode()
    assert len(data) < plan['max_output_bytes']
    assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file()) + len(data) < 1024**3
    assert shutil.disk_usage(ROOT).free >= 40*1024**3
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_bytes(data)
    print(json.dumps({k:result[k] for k in ['state','episodes','physics_steps','optimizer_updates',
          'mixed_groups','mixed_groups_success_ranked_highest','max_analytic_return_error']}),flush=True)


if __name__ == '__main__':
    main()
