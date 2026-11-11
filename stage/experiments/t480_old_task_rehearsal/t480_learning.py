"""Fresh recurrent PPO with an optional task-balanced V-trace replay auxiliary.

One finite pilot proposal. Standard V-trace-inspired adaptation, not full CLEAR
and not an empirical/per-task safety guarantee. No success filtering of replay.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t479_balanced_success_sil'))
import t479_learning as H

C, P, E, M, R, torch, np = H.C, H.P, H.E, H.M, H.R, H.torch, H.np
read, save, sha, digest = H.read, H.save, H.sha, H.digest
make = H.make


def vtrace(reward, discounts, value, log_ratio):
    """Single complete finite episode; rho_bar=c_bar=rho_pg_bar=1.

    value includes one terminal bootstrap, which MUST be zero. The targets
    and policy-advantage weights are detached; gradients use log pi and V only.
    """
    assert reward.ndim == discounts.ndim == log_ratio.ndim == 1
    assert value.shape == (len(reward)+1,) and value[-1] == 0
    assert discounts[-1] == 0 and torch.isfinite(log_ratio).all()
    with torch.no_grad():
        weight = torch.exp(torch.clamp(log_ratio, max=0.))
        delta = weight * (reward + discounts * value[1:] - value[:-1])
        correction = torch.zeros_like(reward[0])
        target = torch.empty_like(reward)
        for i in range(len(reward)-1, -1, -1):
            correction = delta[i] + discounts[i] * weight[i] * correction
            target[i] = value[i] + correction
        next_target = torch.cat((target[1:], value[-1:]))
        advantage = weight * (reward + discounts * next_target - value[:-1])
    return target, advantage, weight


def references(p, own_rows, replay_rows):
    refs = H.references(p, own_rows)
    q = make(p)[0]
    device = next(q.parameters()).device
    actual = []
    with torch.no_grad():
        for row in replay_rows:
            assert sha(row['path']) == row['sha256']
            a = E.N.S.arrays(row['path'])
            x = torch.tensor(a['states'], device=device)
            actual.append(dict(id=row['job_id'], x=x, target=q.sequence(x).detach().clone(),
                               source_sha256=row['sha256']))
    assert len(actual) == len(set(g['id'] for g in actual)) == 8
    refs['actual_old'] = actual
    refs['provenance']['actual_old'] = [{k:g[k] for k in ['id','source_sha256']} for g in actual]
    refs['provenance']['old_weighting'] = 'old211_rotating:.5,actual_old8:.5 within old half;own8 balanced halves'
    return refs


def keep_loss(q, refs, k):
    _, old, own = H.keep_loss(q, refs, k)
    sd = next(q.parameters()).new_tensor(q.p['raw_sigma'])
    current = sum(.5*((q.sequence(g['x'])-g['target'])/sd).square().mean()
                  for g in refs['actual_old'])/8
    return .5*(.5*(old+current)+own), old, current, own


def memory(p, replay_rows, device):
    out = []
    for r in replay_rows:
        assert sha(r['path']) == r['sha256'] and r['full']
        a = E.N.S.arrays(r['path'])
        assert not r['administrative_censor'] and a['stochastic'].all() and not a['ownership'].any()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert np.array_equal(a['states'][1:], a['next_states'][:-1])
        discounts = p['gamma_native']**a['native_counts'].astype(np.int64)
        discounts[-1] = 0.
        out.append(dict(id=r['job_id'], source_sha256=r['sha256'],
                        x=torch.tensor(a['states'],device=device),
                        raw=torch.tensor(a['raw_actions'],device=device),
                        atom=torch.tensor(a['atom'],device=device),
                        behavior=torch.tensor(a['logp'],device=device),
                        reward=torch.tensor(a['rewards'],device=device),
                        discounts=torch.tensor(discounts,device=device),
                        locations=torch.tensor(a['locations'],device=device),
                        terminal=[bool(x) for x in a['end_flags'][-1]]))
    assert len(out) == 8 and len({b['id'] for b in out}) == 8
    return out


def replay_losses(q,v,data,p,assert_behavior=False):
    actor = next(q.parameters()).new_zeros(())
    critic = next(v.parameters()).new_zeros(())
    stats = []
    for b in data:
        loc = q.sequence(b['x'])
        lp = P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])
        if assert_behavior:
            assert float((loc-b['locations']).abs().max()) < 1e-10
            assert float((lp-b['behavior']).abs().max()) < 1e-9
        vv = v(b['x'])
        ext = torch.cat((vv,vv.new_zeros(1)))
        target,advantage,weight = vtrace(b['reward'],b['discounts'],ext,lp-b['behavior'])
        actor = actor - (lp*advantage).mean()/8
        critic = critic + .5*(vv-target).square().mean()/8
        stats.append(dict(id=b['id'],terminal=b['terminal'],
                          clipped_fraction=float(((lp-b['behavior']).detach()>0).double().mean()),
                          weight_mean=float(weight.mean()),weight_min=float(weight.min()),
                          advantage_mean=float(advantage.mean()),
                          advantage_RMS=float(advantage.square().mean().sqrt())))
    return actor,critic,stats


def epoch(obj,items,refs,data,p,k,arm,guard,event):
    guard()
    q,v,ao,vo=obj
    ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True)
    pg=vl=0.
    for b in items:
        ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4
        loss.backward();pg+=float(loss.detach())
        loss=.5*(v(b['x'])-b['target']).square().mean()/4
        loss.backward();vl+=float(loss.detach())
    keep,old,actual,own=keep_loss(q,refs,k)
    (p['keep_weight']*keep).backward()
    ra=rv=0.;stats=[]
    if arm=='B':
        al,cl,stats=replay_losses(q,v,data,p)
        (p['replay_actor_weight']*al).backward()
        (p['replay_value_weight']*cl).backward()
        ra,rv=float(al.detach()),float(cl.detach())
    else:assert arm=='A'
    an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
    vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
    event(arm,'BEFORE',k);ao.step();vo.step();event(arm,'AFTER',k)
    with torch.no_grad():
        kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/4
    return dict(epoch=k,policy_loss=pg,value_loss=vl,keep=float(keep.detach()),
                old211_keep=float(old.detach()),actual_old8_keep=float(actual.detach()),
                own8_keep=float(own.detach()),replay_actor=ra,replay_value=rv,
                replay_stats=stats,actor_norm=an,value_norm=vn,KL=kl)


def fit(obj,arrays,p,refs,data,guard,event,steps=None):
    """The registered capacity-bounded pilot uses ONE proposal, not A/B."""
    assert digest(P.state(*obj)) == p['initial']['joint_hash']
    items,diagnostics=C.batch(obj,arrays,p)
    replay_losses(obj[0],obj[1],data,p,assert_behavior=True)
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        trace.append(epoch(obj,items,refs,data,p,k,'B',guard,event))
        if trace[-1]['KL']>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diagnostics,
                old_replay_NOT_fresh_PPO=True,all_old_outcomes_included=True,
                single_pilot_NOT_matched_ablation=True)
