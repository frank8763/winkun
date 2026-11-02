"""Read-only membership and exact original T302 RNG-index bookkeeping.

No Actor construction, optimizer, simulator or new sampling seed. Reconstruct
only the already consumed first50 index draws specified by T302's source.
"""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'stage/docs/distance_20261102/T303_RETENTION_SAMPLING.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    protocol = ROOT / 'stage/runs/t302_balanced_bridge_learning/protocol.json'
    p = json.loads(protocol.read_text())
    summary = json.loads((protocol.parent / 'summary.json').read_text())
    assert summary['supervised']['attempted_steps'] == 50
    lengths = []
    for source in p['auxiliary_sources']:
        assert sha(source['path']) == source['sha256']
        with np.load(source['path']) as z:
            lengths.append(len(z['states']))
    assert sha(p['retention_states']) == p['retention_states_sha256']
    with np.load(p['retention_states']) as z:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in z['case_ids']]
        offsets, tasks = z['offsets'].copy(), z['task'].copy()
    critical = ids.index('dev_task6_24')
    group = int(tasks[critical])
    base = sum(int(offsets[k+1] - offsets[k]) for k in range(len(ids)) if tasks[k] == group and k < critical)
    end = base + int(offsets[critical+1] - offsets[critical])
    total = sum(int(offsets[k+1] - offsets[k]) for k in range(len(ids)) if tasks[k] == group)
    draws = []
    for step in range(50):
        rng = np.random.default_rng(p['supervised_seed'] + step)
        for length in lengths:
            rng.integers(0, length, 4)  # Original15 auxiliary draws occur first.
        for task in range(8):
            size = sum(int(offsets[k+1] - offsets[k]) for k in range(len(ids)) if tasks[k] == task)
            ix = rng.integers(0, size, 4)
            if task == group:
                selected = ix[(ix >= base) & (ix < end)] - base
                draws.append(dict(step=step, critical_local_indices=selected.tolist()))
    selected = [i for d in draws for i in d['critical_local_indices']]
    result = dict(scope='Posthoc exact original fixed50 RNG index bookkeeping, zero new data/optimizer/Actor/native',
        retention_case_count=len(ids), critical_id=ids[critical], critical_rows=end-base,
        group_rows=total, original50_total_keep_draws=50*8*4,
        critical_keep_draws=len(selected), critical_distinct_rows=len(set(selected)),
        hold06_in_retention='hold_task6_06' in ids, original_seed=p['supervised_seed'],
        draws=draws, protocol_sha256=sha(protocol), retention_sha256=sha(p['retention_states']),
        source_sha256=sha(__file__), native_steps=0, optimizer_calls=0,
        caveat='dev24 is already in the full211 replay; do not claim missing data. Task-balanced random row sampling is not equal-case or worst-case protection. Counts do not establish the cause of collision or prove replay reweighting will fix6m. The consumed hold06 guard is not a new untouched evaluation family.')
    DEST.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='draws'}))


if __name__ == '__main__':
    main()
