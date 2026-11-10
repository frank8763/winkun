"""Fixed night recovery objectives; supervised labels never become PPO data.

Occupancy emphasizes the actual student prefix and first teacher decision.
Duration instead weights every decision by its actual native physical count.
Both retain the input policy function on known-success historical support.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'stage468_pinned465', ROOT/'stage/experiments/t465_matched_safe_recovery/run.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
Y, L = V.Y, V.L
X, J, H, M, P, E, D, F, R = V.X, V.J, V.H, V.M, V.P, V.E, V.D, V.F, V.R
torch, np = V.torch, V.np
read, save, sha, digest = V.read, V.save, V.sha, V.digest
state, load, export, frozen, base_digest = Y.state, Y.load, Y.export, Y.frozen, Y.base_digest
MODES = ('occupancy', 'duration')


def make(p, device='cuda'):
    obj = Y.make(p, device)
    q = obj[0]
    assert sum(v.numel() for v in q.parameters() if v.requires_grad) == 1074
    assert sum(v.numel() for v in q.parameters() if not v.requires_grad) == 52866
    return obj


def _arrays(row, safe=False):
    assert row['full'] and sha(row['path']) == row['sha256']
    a = E.N.S.arrays(row['path'])
    assert digest(a) == row['array_digest']
    assert {k:digest(v) for k,v in a.items()} == row['array_hashes']
    assert a['states'].ndim == 2 and a['states'].shape[1] == 120
    assert np.array_equal(a['states'][1:],a['next_states'][:-1])
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
    assert [bool(v) for v in a['end_flags'][-1]] == [row['collision'],row['success'],row['timeout']]
    assert not row.get('administrative_censor',False)
    for value in a.values():
        if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all()
    if safe:assert row['success'] and not row['collision'] and not row['timeout']
    return a


def data(p, student_rows, recovery_rows, mode, old_recoveries=None, device='cuda'):
    assert mode in MODES and len(recovery_rows) <= 2
    valid = set(p['current_success_ids'])
    anchor = frozen(p,p['learning_initial']['path']).to(device)
    groups = L._anchors(anchor,[g for g in M.dataset(p) if g['id'] in valid],device)
    old = [r for r in p['own_rows']+p['new_original_own5'] if r['job_id'] in valid]
    histories, current = [], []
    for row in student_rows:
        a = _arrays(row)
        assert not row['teacher'] and not a['ownership'].any()
        assert row['policy_hash'] == p['learning_initial']['policy_hash']
        if row['success']:
            assert row['job_id'] in valid
            current.append(row)
    for row in old+current:
        a = _arrays(row,safe=True)
        assert not row['teacher'] and not a['ownership'].any()
        histories.append(dict(id=row['job_id'],task=10,x=a['states'],y=a['actions']))
    own = L._anchors(anchor,histories,device)
    rows, seen = [], set()
    for row in list(recovery_rows)+list(old_recoveries or []):
        key = (row['path'],row['sha256'])
        if key not in seen:rows.append(row);seen.add(key)
    assert 1 <= len(rows) <= 4
    corrections, provenance = [], []
    for row in rows:
        a = _arrays(row,safe=True)
        first = row['teacher_first']
        assert row['teacher'] and isinstance(first,int) and 0 <= first < len(a['states'])
        assert not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert a['teacher_actions'].dtype == np.float32
        assert a['teacher_actions'].shape == a['actions'].shape == (len(a['states']),2)
        counts = np.asarray(a['native_counts'],dtype=np.float64)
        assert counts.ndim == 1 and len(counts) == len(a['states']) and (counts>0).all()
        corrections.append(dict(id=row['job_id'],x=torch.tensor(a['states'],device=device),
            y=torch.tensor(a['teacher_actions'],device=device,dtype=torch.float64),
            native_weight=torch.tensor(counts/counts.sum(),device=device,dtype=torch.float64),
            first=first,prefix_decisions=first+1))
        provenance.append(dict(id=row['job_id'],path=row['path'],sha256=row['sha256'],
            source_policy_hash=row['policy_hash'],input_policy_hash=p['learning_initial']['policy_hash'],
            source_policy_matches_input=row['policy_hash']==p['learning_initial']['policy_hash'],
            teacher_first=first,decisions=len(counts),physical_steps=int(counts.sum()),
            old_support=row not in recovery_rows,supervised_offpolicy_NOT_probability_ratio_data=True))
    assert all(g['id'] in valid for g in groups+own)
    return dict(mode=mode,groups=groups,own=own,corrections=corrections,
        reference_count=len(groups),old_own_count=len(old),current_own_count=len(current),
        own_count=len(own),correction_count=len(corrections),correction_provenance=provenance,
        current_own_support_paths=[r['path'] for r in current],
        retained_functions_are_input_model_on_finite_support_NOT_global_safety=True)


def _retention(q, batch):
    reference = own_loss = 0.
    for task in range(10):
        selected = [g for g in batch['groups'] if g['task'] == task]
        weight = 1/16 if task < 8 else 1/4
        for off in range(0,len(selected),8):
            chunk = selected[off:off+8]
            x = torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk])
            target = torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk])
            loc = q.sequence(x)
            loss = sum((loc[:len(g['x']),i]-target[:len(g['x']),i]).square().mean()
                for i,g in enumerate(chunk))*weight/len(selected)
            loss.backward();reference += float(loss.detach())
    for off in range(0,len(batch['own']),5):
        chunk = batch['own'][off:off+5]
        x = torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk])
        target = torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk])
        loc = q.sequence(x)
        loss = sum((loc[:len(g['x']),i]-target[:len(g['x']),i]).square().mean()
            for i,g in enumerate(chunk))/len(batch['own'])
        loss.backward();own_loss += float(loss.detach())
    return reference,own_loss


def bc_step(obj, batch, mode=None):
    mode = batch['mode'] if mode is None else mode
    assert mode == batch['mode'] and mode in MODES
    q,opt = obj
    assert q._t463_mode == 'aggregate'
    L._assert_frozen(q);opt.zero_grad(set_to_none=True)
    result = dict(mode=mode,student_prefix=0.,teacher_suffix=0.,duration_loss=0.,
        correction_objective=0.,reference218=0.,own10=0.)
    count = len(batch['corrections'])
    for g in batch['corrections']:
        loc = q.sequence(g['x'])
        loss_per_axis = P.action_loss(loc,g['y'])
        n = g['prefix_decisions']
        early = loss_per_axis[:n].mean()
        suffix = loss_per_axis[n:].mean() if n < len(loc) else loc.sum()*0.
        duration = (loss_per_axis.mean(dim=1)*g['native_weight']).sum()
        loss = early+.1*suffix if mode == 'occupancy' else duration
        (loss/count).backward()
        result['student_prefix'] += float(early.detach())/count
        result['teacher_suffix'] += float(suffix.detach())/count
        result['duration_loss'] += float(duration.detach())/count
        result['correction_objective'] += float(loss.detach())/count
    result['reference218'],result['own10'] = _retention(q,batch)
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(L._trainable(q),1.,error_if_nonfinite=True))
    L._assert_frozen(q);opt.step();L._assert_frozen(q)
    return result
