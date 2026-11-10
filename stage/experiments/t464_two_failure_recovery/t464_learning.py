"""One fixed two-failure supervised recovery update on the frozen-base Actor.

The input may include the previous PPO Value/optimizer state. That state stays
in its original archived checkpoint; this module restores only Actor/residual
Adam and performs no Value or new reinforcement-learning updates.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'stage464_pinned463_run', ROOT/'stage/experiments/t463_evening_2000/run.py')
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
L, J = X.L, X.J
H, M, P, E, D, F, R = X.H, X.M, X.P, X.E, X.L.D, X.J.F, X.R
torch, np = X.torch, X.np
read, save, sha, digest = X.read, X.save, X.sha, X.digest
state, load, export, frozen, base_digest = L.state, L.load, L.export, L.frozen, L.base_digest


def make(p, device='cuda'):
    # L.make checks the complete source checkpoint joint hash, restores the
    # mature residual Adam group, and verifies the unchanged deployment policy.
    return L.make(p, 'aggregate', device)


def _safe_arrays(row):
    assert row['success'] and not row['collision'] and not row['timeout']
    assert not row.get('administrative_censor', False)
    assert row.get('full', True) and sha(row['path']) == row['sha256']
    a = E.N.S.arrays(row['path'])
    assert digest(a) == row['array_digest']
    assert {k: digest(v) for k, v in a.items()} == row['array_hashes']
    assert not a['end_flags'][:-1].any()
    assert a['end_flags'][-1].tolist() == [0, 1, 0]
    assert a['states'].ndim == 2 and a['states'].shape[1] == 120
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    return a


def data(p, new_own4, recovery2, device='cuda'):
    assert len(new_own4) == 4 and len(recovery2) == 2
    assert len({r['job_id'] for r in new_own4}) == 4
    assert len({r['job_id'] for r in recovery2}) == 2
    valid = set(p['current_success_ids'])
    assert all(r['job_id'] in valid for r in new_own4)
    anchor = frozen(p, p['learning_initial']['path']).to(device)
    groups = L._anchors(anchor, [g for g in M.dataset(p) if g['id'] in valid], device)
    old = [r for r in p['own_rows']+p['new_original_own5'] if r['job_id'] in valid]
    histories = []
    for row in old+new_own4:
        a = _safe_arrays(row)
        assert not row['teacher'] and not a['ownership'].any()
        histories.append(dict(id=row['job_id'], task=10, x=a['states'], y=a['actions']))
    own = L._anchors(anchor, histories, device)
    for row in recovery2:
        a = _safe_arrays(row)
        assert row['teacher'] and row['teacher_first'] == 20
        assert len(a['states']) > 21 and 'teacher_actions' in a
        assert not a['ownership'][:20].any() and a['ownership'][20:].all()
    corrections = [{k: v.to(device) for k, v in D.correction(row).items()} for row in recovery2]
    assert all(g['id'] in valid for g in groups+own)
    return dict(groups=groups, own=own, corrections=corrections,
                success_ids=sorted(valid), reference_count=len(groups),
                old_own_count=len(old), own_count=len(own), correction_count=2)


def bc_step(obj, batch):
    return L.bc_step(obj, batch, 'aggregate')
