"""Matched recurrent PPO arms; optional, explicitly off-policy success SIL.

Both arms share initial mature Actor/Value Adam, fresh PPO batch, complete
success histories and epoch count. A shared KL stop halts BOTH arms together.
No optimizer step is executed by importing this module.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t478_wide8_pooled_credit'))
import t478_learning as C

P, E, M, R, torch, np = C.P, C.E, C.M, C.R, C.torch, C.np
read, save, sha, digest = C.read, C.save, C.sha, C.digest
make = C.make


def side(name):
    assert ('_LEFT_' in name) != ('_RIGHT_' in name), name
    return 'LEFT' if '_LEFT_' in name else 'RIGHT'


def returns(a, gamma):
    """True finite-cap MC return; no administrative-censor bootstrap fiction."""
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
    out = np.empty(len(a['rewards']), np.float64)
    value = 0.
    for i in reversed(range(len(out))):
        value = float(a['rewards'][i]) + gamma ** int(a['native_counts'][i]) * value
        out[i] = value
    return out


def references(p, own_rows):
    # Retain the old211 schedule, replacing the incomplete own-history list.
    pp = dict(p, own_reference_rows=[])
    refs = C.references(pp, own_rows)
    assert len(own_rows) == len(set(r['job_id'] for r in own_rows)) == 8
    assert set(r['job_id'] for r in own_rows) == set(p['capture_success_ids'])
    for g in refs['own']:
        g['side'] = side(g['id'])
    assert {g['side'] for g in refs['own']} == {'LEFT', 'RIGHT'}
    refs['provenance']['side_total_weights'] = dict(LEFT=.5, RIGHT=.5)
    return refs


def keep_loss(q, refs, step):
    pools = [[g for g in refs['old'] if g['task'] == t] for t in range(8)]
    selected = [pool[step % len(pool)] for pool in pools]
    x = torch.nn.utils.rnn.pad_sequence([g['x'] for g in selected])
    loc = q.sequence(x)
    sd = loc.new_tensor(q.p['raw_sigma'])
    old = sum(.5 * ((loc[:len(g['x']), i] - g['target']) / sd).square().mean()
              for i, g in enumerate(selected)) / 8
    own = loc.new_zeros(())
    for which in ('LEFT', 'RIGHT'):
        group = [g for g in refs['own'] if g['side'] == which]
        assert group
        own = own + sum(.5 * ((q.sequence(g['x']) - g['target']) / sd).square().mean()
                        for g in group) / (2 * len(group))
    return .5 * (old + own), old, own


def success_memory(p, device):
    memory = []
    for row in p['sil_rows']:
        assert sha(row['path']) == row['sha256']
        a = E.N.S.arrays(row['path'])
        assert row['success'] and not row['collision'] and not row['administrative_censor']
        assert a['end_flags'][-1, 1] and not a['end_flags'][:, 0].any()
        assert a['stochastic'].all() and not a['ownership'].any()
        assert np.array_equal(a['states'][1:], a['next_states'][:-1])
        memory.append(dict(id=row['job_id'], side=side(row['job_id']),
                           source_sha256=row['sha256'],
                           x=torch.tensor(a['states'], device=device),
                           raw=torch.tensor(a['raw_actions'], device=device),
                           atom=torch.tensor(a['atom'], device=device),
                           G=torch.tensor(returns(a, p['gamma_native']), device=device)))
    assert len(memory) == len(set(g['source_sha256'] for g in memory))
    assert {g['side'] for g in memory} == {'LEFT', 'RIGHT'}
    return memory


def sil_losses(q, v, memory, p):
    actor = next(q.parameters()).new_zeros(())
    value = next(v.parameters()).new_zeros(())
    positive = {}
    for which in ('LEFT', 'RIGHT'):
        group = [g for g in memory if g['side'] == which]
        assert group
        for g in group:
            advantage = (g['G'] - v(g['x'])).clamp_min(0.)
            # Stop gradient through Actor weights; Value has its own loss.
            logp = P.L.density(q.sequence(g['x']), g['raw'], g['atom'], p['raw_sigma'])
            actor = actor - (logp * advantage.detach()).mean() / (2 * len(group))
            value = value + .5 * advantage.square().mean() / (2 * len(group))
            positive[g['source_sha256']] = float((advantage.detach() > 0).double().mean())
    return actor, value, positive


def epoch(obj, items, refs, memory, p, k, arm, guard, event):
    guard()
    q, v, ao, vo = obj
    ao.zero_grad(set_to_none=True)
    vo.zero_grad(set_to_none=True)
    pg = vl = 0.
    for b in items:
        ratio = (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma']) - b['old']).exp()
        loss = -torch.minimum(ratio * b['adv'], ratio.clamp(.8, 1.2) * b['adv']).mean() / 4
        loss.backward()
        pg += float(loss.detach())
        loss = .5 * (v(b['x']) - b['target']).square().mean() / 4
        loss.backward()
        vl += float(loss.detach())
    keep, old, own = keep_loss(q, refs, k)
    (p['keep_weight'] * keep).backward()
    sa = sv = 0.
    positive = {}
    if arm == 'B':
        al, value, positive = sil_losses(q, v, memory, p)
        (p['sil_actor_weight'] * al).backward()
        (p['sil_value_weight'] * value).backward()
        sa, sv = float(al.detach()), float(value.detach())
    else:
        assert arm == 'A'
    an = float(torch.nn.utils.clip_grad_norm_(q.parameters(), 1., error_if_nonfinite=True))
    vn = float(torch.nn.utils.clip_grad_norm_(v.parameters(), 1., error_if_nonfinite=True))
    event(arm, 'BEFORE', k)
    ao.step()
    vo.step()
    event(arm, 'AFTER', k)
    with torch.no_grad():
        kl = sum(float(((z := P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])
                         - b['old']).exp() - 1 - z).mean()) for b in items) / 4
    return dict(epoch=k, policy_loss=pg, value_loss=vl, keep=float(keep.detach()),
                old_keep=float(old.detach()), own_keep=float(own.detach()),
                SIL_actor=sa, SIL_value=sv, SIL_positive_fraction=positive,
                actor_norm=an, value_norm=vn, KL=kl)


def fit_pair(objects, arrays, p, refs, memory, guard, event, steps=None):
    assert set(objects) == {'A', 'B'}
    assert digest(P.state(*objects['A'])) == digest(P.state(*objects['B'])) == p['initial']['joint_hash']
    batches = {arm: C.batch(obj, arrays, p) for arm, obj in objects.items()}
    assert batches['A'][1] == batches['B'][1]
    trace = {arm: [] for arm in objects}
    for k in range(p['epochs'] if steps is None else steps):
        for arm, obj in objects.items():
            trace[arm].append(epoch(obj, batches[arm][0], refs, memory, p, k, arm, guard, event))
        # Match realized PPO/Adam count even when just one arm exceeds KL.
        if max(trace[arm][-1]['KL'] for arm in objects) > p['target_kl']:
            break
    return dict(trace=trace, matched_epochs=len(trace['A']),
                pooled_advantage=batches['A'][1],
                shared_KL_stop=True, historical_SIL_NOT_onpolicy_PPO=True)
