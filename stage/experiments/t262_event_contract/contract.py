"""First-event labels and probability identities only; no learned controller."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t262_event_contract'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def targets(counts, flags, cap, complete, horizons):
    counts = np.asarray(counts, dtype=np.int64)
    flags = np.asarray(flags, dtype=np.uint8)
    assert len(counts) and flags.shape == (len(counts), 3)
    assert np.all(counts > 0) and np.all(counts <= 100)
    assert np.all(flags[:-1] == 0) and int(flags[-1].sum()) <= 1
    n = int(counts.sum())
    assert n <= cap
    goal, contact, timeout = map(bool, flags[-1])
    assert bool(goal or contact or timeout) == bool(complete)
    if timeout:
        assert n == cap
    if not complete:
        assert n < cap
    event = 0 if goal else 1 if contact else None
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    horizons = np.asarray(horizons, dtype=np.int64)
    assert np.all(horizons > 0)
    mask = np.zeros((len(counts), len(horizons)), dtype=np.uint8)
    labels = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, start in enumerate(starts):
        remaining_observed = n - start
        remaining_cap = cap - start
        for j, horizon in enumerate(horizons):
            if horizon > remaining_cap:
                continue
            if event is not None and remaining_observed <= horizon:
                mask[i, j] = 1
                labels[i, j, event] = 1
            elif horizon <= remaining_observed:
                mask[i, j] = 1
                labels[i, j, 2] = 1
            # Unobserved suffix after administrative censoring stays masked.
    assert np.array_equal(labels.sum(-1), mask)
    return labels, mask


def probabilities(hazards):
    h = np.asarray(hazards, dtype=np.float64)
    assert h.ndim == 2 and h.shape[1] == 2
    assert np.isfinite(h).all() and np.all(h >= 0) and np.all(h.sum(-1) <= 1)
    survival = np.r_[1., np.cumprod(1-h.sum(-1))]
    cumulative = np.cumsum(survival[:-1, None]*h, axis=0)
    mass = cumulative.sum(-1) + survival[1:]
    assert np.allclose(mass, 1, atol=1e-14, rtol=0)
    assert np.all(np.diff(cumulative, axis=0) >= 0)
    return cumulative, survival[1:]


def synthetic_contracts():
    h = [50, 100, 150, 200]
    g, m = targets([100, 50], [[0,0,0],[0,1,0]], 200, True, h)
    assert g[0,2,0] == g[0,3,0] == g[1,0,0] == 1
    assert g[0,0,2] == 1 and m[1,2] == 0
    c, _ = targets([100,50], [[0,0,0],[1,0,0]], 200, True, h)
    assert c[0,2,1] == 1 and c[...,0].sum() == 0
    t, tm = targets([100,100], [[0,0,0],[0,0,1]], 200, True, h)
    assert t[0,3,2] == 1 and tm[1,2] == 0
    w, wm = targets([100], [[0,0,0]], 300, False, h)
    assert w[0,1,2] == 1 and wm[0,2] == wm[0,3] == 0
    f, s = probabilities([[.2,.1],[.1,.2],[.4,.1]])
    assert np.max(np.abs(f.sum(-1)+s-1)) < 1e-14
    f0, s0 = probabilities([[0.,0.],[0.,0.]])
    f1, s1 = probabilities([[1.,0.],[0.,0.]])
    assert f0.sum() == 0 and np.all(s0 == 1)
    assert np.all(f1[:,0] == 1) and np.all(s1 == 0)
    return dict(goal_contact_exclusive=True, physical_time_boundaries=True,
        true_timeout_observed_no_event=True, administrative_censoring_unknown_suffix=True,
        original_cap_not_extrapolated=True, probability_mass_conserved=True,
        identity_is_standard_not_novel=True)


def main():
    start = time.monotonic()
    plan = json.loads((HERE/'protocol.json').read_text())
    assert not (OUT/'summary.json').exists(), 'No repeat/enlarged data prefix'
    assert sha(Path(__file__).read_bytes()) == plan['source_sha256']
    checks = synthetic_contracts()
    lines = Path(plan['source_rows']).read_bytes().splitlines(keepends=True)[:48]
    assert len(lines) == 48 and sha(b''.join(lines)) == plan['source_prefix_sha256']
    all_labels, all_masks, offsets, records = [], [], [0], []
    for line in lines:
        assert time.monotonic()-start < plan['wall_seconds']
        row = json.loads(line)
        path = Path(row['path'])
        assert sha(path.read_bytes()) == row['sha256']
        with np.load(path) as z:
            count, flag = z['native_counts'], z['end_flags']
            actor_states_available = 'states' in z.files
        assert not actor_states_available, 'Do not silently change the data-availability contract'
        labels, mask = targets(count, flag, row['job']['case']['step_cap'],
                               row['complete'], plan['horizons_physics_steps'])
        assert int(count.astype(np.int64).sum()) == row['physics_steps']
        all_labels.append(labels)
        all_masks.append(mask)
        offsets.append(offsets[-1]+len(count))
        records.append(dict(id=row['id'], safe=bool(row['success'] and not row['collision']),
            contact=bool(row['collision']), true_timeout=bool(row['timeout']),
            historical_overlap=bool(row.get('exact_historical_overlap',False)),
            decisions=len(count), valid_horizon_labels=int(mask.sum())))
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, labels=labels, valid=masks,
                        offsets=np.array(offsets,dtype=np.int64),
                        horizons=np.array(plan['horizons_physics_steps'],dtype=np.int64))
    payload = buffer.getvalue()
    result = dict(state='COMPLETE', physics_steps=0, optimizer_updates=0,
        learned_critic=False, Actor_changed=False, actual_navigation_success_added=False,
        synthetic_contracts=checks, episodes=48, decision_rows=len(labels),
        horizons_seconds=plan['horizons_seconds'], valid_labels=int(masks.sum()),
        goal_labels=int(labels[...,0].sum()), contact_labels=int(labels[...,1].sum()),
        observed_no_event_labels=int(labels[...,2].sum()),
        masked_unobserved_or_beyond_cap=int(masks.size-masks.sum()),
        episodes_detail=records, independent_sample_count_not_decision_count=True,
        critic_training_data_ready=False,
        missing='T259 compact episode files omit 120D states. New learned critic requires '
                'separately bounded exact recovery of states and causal goal/history relabeling '
                'contracts; labels alone do not provide training data or calibrated risk.',
        probability_identity_not_safety_certificate=True, elapsed_s=time.monotonic()-start,
        labels_sha256=sha(payload), labels_bytes=len(payload),
        original_T259_protocol_unchanged=True)
    report = (json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    assert len(report)+len(payload) < plan['max_output_bytes']
    used = sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+len(report)+len(payload) < 1024**3 and shutil.disk_usage(ROOT).free >= 40*1024**3
    (OUT/'labels.npz').write_bytes(payload)
    (OUT/'summary.json').write_bytes(report)
    print(json.dumps({k:result[k] for k in ['state','physics_steps','optimizer_updates',
          'episodes','decision_rows','valid_labels','critic_training_data_ready']}),flush=True)


if __name__ == '__main__':
    main()
