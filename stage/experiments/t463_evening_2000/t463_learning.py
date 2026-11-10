"""Finite residual learning engines; deployment retains the existing Actor.

The bounded modes freeze the existing 64->16 residual features. Projection
changes output parameters after Adam, and leaves every Adam moment untouched.
"""
from pathlib import Path
import copy
import gzip
import importlib.util
import io
import math

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'stage463_pinned462', ROOT/'stage/experiments/t462_recurrent_parent_correction/run.py')
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)
H, M, P, D, E, torch, np = J.H, J.M, J.P, J.D, J.E, J.torch, J.np
read, sha, digest = J.read, J.sha, J.digest
MODES = ('aggregate', 'bounded', 'ppo')
LOGIT_BOUND = .016
LOCATION_BOUND = .02
base_digest = H.base_digest
frozen = H.frozen


def _trainable(q):
    return [v for v in q.parameters() if v.requires_grad]


def _fixed_digest(q):
    return digest({n: v for n, v in q.state_dict().items()
                   if not n.startswith('head.residual.2.')})


def make(p, mode, device='cuda'):
    assert mode in MODES
    q, opt = H.make(p, device)
    c = P.checkpoint(p['learning_initial']['path'])
    assert digest(c['residual_state']) == c['joint_hash'] == p['learning_initial']['joint_hash']
    H.load((q, opt), c['residual_state'])
    assert q.phash() == p['learning_initial']['policy_hash']
    q._t463_mode = mode
    q._t463_initial_output = [v.detach().clone() for v in q.head.residual[-1].parameters()]
    if mode != 'aggregate':
        q.head.residual[0].requires_grad_(False)
    # Keep the complete mature residual optimizer parameter group and moments.
    assert sum(v.numel() for v in _trainable(q)) == (1074 if mode == 'aggregate' else 34)
    assert sum(v.numel() for v in q.parameters()) == 52866+1074
    assert base_digest(q) == p['base_actor_digest']
    q._t463_fixed_digest = _fixed_digest(q)
    if mode != 'ppo':
        return q, opt
    v = P.L.Value(q, p['value_seed']).to(device)
    v.core.train()
    vo = torch.optim.Adam(v.parameters(), lr=p['value_lr'], foreach=False, fused=False)
    return q, opt, v, vo


def state(q, opt, value=None, value_opt=None):
    result = dict(actor=q.state_dict(), residual_opt=opt.state_dict())
    if value is not None:
        assert value_opt is not None
        result.update(value=value.state_dict(), value_opt=value_opt.state_dict())
    return P.L.cpu(result)


def load(obj, saved):
    H.load(obj[:2], saved)
    if len(obj) == 4 and 'value' in saved:
        obj[2].load_state_dict(saved['value'])
        obj[3].load_state_dict(copy.deepcopy(saved['value_opt']))
    _assert_frozen(obj[0])


def export(obj, path, limit):
    s = state(*obj)
    buf = io.BytesIO()
    torch.save(dict(residual_state=s, architecture=H.ARCH, policy_hash=obj[0].phash(),
                    joint_hash=digest(s), single_actor=True,
                    frozen_parent=obj[0].p['research_initial']), buf)
    raw = buf.getvalue()
    assert len(raw)+65536 <= limit, (len(raw), limit)
    blob = gzip.compress(raw, mtime=0)
    assert len(blob) <= limit and not Path(path).exists()
    Path(path).write_bytes(blob)
    return dict(path=str(path), sha256=sha(path), policy_hash=obj[0].phash(),
                joint_hash=digest(s), bytes=len(blob), uncompressed_bytes=len(raw))


def _anchors(q, groups, device):
    result = []
    with torch.no_grad():
        for g in P.packs(groups, device):
            result.append(dict(g, target=q.sequence(g['x']).detach().clone()))
    return result


def build_constraint(anchor, groups, own):
    """Full causal support from h0, with frozen base and first residual layer."""
    features, logits, episodes = [], [], []
    with torch.no_grad():
        for g in groups+own:
            x = g['x'][:, None]
            hidden, _ = anchor.core(anchor.encoder(anchor.normalized(x)))
            phi = anchor.head.residual[:2](hidden[:, 0])
            features.append(phi.detach())
            logits.append(anchor.head(hidden[:, 0]).detach())
            episodes.append(dict(id=g['id'], decisions=len(x)))
        assert features
        w, b = [v.detach().clone() for v in anchor.head.residual[-1].parameters()]
    return dict(phi=torch.cat(features), initial_logits=torch.cat(logits),
                weight=w, bias=b, episodes=episodes, bound=LOGIT_BOUND,
                fixed_digest=_fixed_digest(anchor))


def build_data(p, device='cuda'):
    anchor = frozen(p, p['learning_initial']['path']).to(device)
    valid = set(p['current_success_ids'])
    refs = [g for g in M.dataset(p) if g['id'] in valid]
    groups = _anchors(anchor, refs, device)
    old = [r for r in p['own_rows']+p['new_original_own5'] if r['job_id'] in valid]
    current = p['current_own6']
    assert len(current) == 6 and all(r['job_id'] in valid for r in current)
    rows = old+current
    histories = []
    for r in rows:
        assert r['success'] and not r['collision'] and sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist() == [0, 1, 0]
        histories.append(dict(id=r['job_id'], task=10, x=a['states'], y=a['actions']))
    own = _anchors(anchor, histories, device)
    corrections = [D.correction(r) for r in p['aggregate_recovery10']]
    corrections = [{k: v.to(device) for k, v in g.items()} for g in corrections]
    assert len(corrections) == 10
    assert all(g['id'] in valid for g in groups+own)
    return dict(groups=groups, own=own, corrections=corrections,
                constraint=build_constraint(anchor, groups, own),
                success_ids=sorted(valid), reference_count=len(groups), own_count=len(own),
                old_own_count=len(old), correction_count=len(corrections))


def _assert_frozen(q):
    assert base_digest(q) == q.p['base_actor_digest']
    if q._t463_mode != 'aggregate':
        assert _fixed_digest(q) == q._t463_fixed_digest
    assert all(v.grad is None for v in q.parameters() if not v.requires_grad)


def constraint_snapshot(q, support):
    with torch.no_grad():
        w, b = q.head.residual[-1].parameters()
        dw, db = w-support['weight'], b-support['bias']
        delta = support['phi']@dw.T+db
        location = 1.25*((support['initial_logits']+delta).tanh()-support['initial_logits'].tanh())
        return dict(logit_abs_max=float(delta.abs().max()),
                    location_abs_max=float(location.abs().max()),
                    location_mean_abs=float(location.abs().mean()),
                    support_decisions=len(delta), support_episodes=len(support['episodes']))


def project(q, support):
    assert q._t463_mode in ('bounded', 'ppo')
    assert _fixed_digest(q) == support['fixed_digest'] == q._t463_fixed_digest
    assert all(torch.equal(a, b) for a, b in
               zip(q._t463_initial_output, (support['weight'], support['bias'])))
    before = constraint_snapshot(q, support)
    scale = 1.
    if before['logit_abs_max'] > LOGIT_BOUND:
        scale = LOGIT_BOUND*(1-1e-12)/before['logit_abs_max']
        with torch.no_grad():
            for parameter, initial in zip(q.head.residual[-1].parameters(), q._t463_initial_output):
                parameter.copy_(initial+(parameter-initial)*scale)
    after = constraint_snapshot(q, support)
    assert after['logit_abs_max'] <= LOGIT_BOUND+1e-12
    assert after['location_abs_max'] <= LOCATION_BOUND+1e-12
    assert after['location_mean_abs'] <= LOCATION_BOUND+1e-12
    return dict(pre=before, post=after, radial_scale=scale)


def bc_step(obj, data, mode):
    q, opt = obj[:2]
    assert mode in ('aggregate', 'bounded') and q._t463_mode == mode
    opt.zero_grad(set_to_none=True)
    result = D.auxiliary_backward(q, data['groups'], data['own'], data['corrections'], 1.)
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
        _trainable(q), 1., error_if_nonfinite=True))
    _assert_frozen(q)
    opt.step()
    if mode == 'bounded':
        result['constraint'] = project(q, data['constraint'])
    _assert_frozen(q)
    return result


def ppo_fit(obj, arrays, p, data, guard, event):
    q, ao, v, vo = obj
    assert q._t463_mode == 'ppo' and len(arrays) == 8
    device = next(q.parameters()).device
    batch, errors = [], []
    for a in arrays:
        assert not np.asarray(a['ownership']).any() and np.asarray(a['stochastic']).all()
        assert np.array_equal(a['states'][1:], a['next_states'][:-1])
        flags = np.asarray(a['end_flags'])
        assert not flags[:-1].any() and flags[-1].sum() == 1
        x = torch.as_tensor(a['states'], device=device)
        raw = torch.as_tensor(a['raw_actions'], device=device)
        atom = torch.as_tensor(a['atom'], device=device)
        old = torch.as_tensor(a['logp'], device=device)
        with torch.no_grad():
            loc = q.sequence(x)
            er = float((loc-loc.new_tensor(a['locations'])).abs().max())
            lp = float((P.L.density(loc, raw, atom, p['raw_sigma'])-old).abs().max())
            assert er < 1e-10 and lp < 1e-9, (er, lp)
            extended = torch.as_tensor(np.concatenate([a['states'], a['next_states'][-1:]]), device=device)
            vv = v(extended).cpu().numpy()
        mask = np.ones((len(x), 1), bool)
        mask[-1] = False
        adv, target = P.L.physical_gae(np.asarray(a['rewards'])[:, None], vv[:-1, None], vv[1:, None],
            np.asarray(a['native_counts'], dtype=np.int64)[:, None], mask, mask,
            gamma_native=p['gamma_native'], lambda_native=p['lambda_native'])
        batch.append(dict(x=x, raw=raw, atom=atom, old=old, adv=adv[:, 0],
                          target=loc.new_tensor(target[:, 0])))
        errors.append([er, lp])
    # Equal episode weighting also defines normalization, independent of length.
    mean = float(np.mean([b['adv'].mean() for b in batch]))
    second = float(np.mean([np.mean(b['adv']**2) for b in batch]))
    sd = math.sqrt(max(0., second-mean**2))+1e-8
    for b in batch:
        b['adv'] = b['old'].new_tensor((b['adv']-mean)/sd)
    epochs = p.get('ppo_epochs', 4)
    assert 1 <= epochs <= 4 and p.get('ppo_aux_weight', .1) == .1
    assert p.get('target_kl', .02) == .02
    trace = []
    for k in range(epochs):
        guard()
        ao.zero_grad(set_to_none=True)
        vo.zero_grad(set_to_none=True)
        pg = vl = 0.
        for b in batch:
            ratio = (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])-b['old']).exp()
            loss = -torch.minimum(ratio*b['adv'], ratio.clamp(.8, 1.2)*b['adv']).mean()/len(batch)
            loss.backward()
            pg += float(loss.detach())
            loss = .5*(v(b['x'])-b['target']).square().mean()/len(batch)
            loss.backward()
            vl += float(loss.detach())
        policy_norm = float(sum(t.grad.square().sum() for t in _trainable(q) if t.grad is not None).sqrt())
        aux = D.auxiliary_backward(q, data['groups'], data['own'], data['corrections'], .1)
        an = float(torch.nn.utils.clip_grad_norm_(_trainable(q), 1., error_if_nonfinite=True))
        vn = float(torch.nn.utils.clip_grad_norm_(v.parameters(), 1., error_if_nonfinite=True))
        _assert_frozen(q)
        event('BEFORE', k)
        ao.step()
        bound = project(q, data['constraint'])
        vo.step()
        event('AFTER', k)
        _assert_frozen(q)
        with torch.no_grad():
            kl = sum(float(((z := (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])-b['old'])).exp()-1-z).mean())
                     for b in batch)/len(batch)
        assert math.isfinite(kl)
        trace.append(dict(epoch=k, policy_loss=pg, value_loss=vl,
                          policy_only_gradient_norm=policy_norm, auxiliary=aux,
                          actor_norm=an, value_norm=vn, KL=kl, constraint=bound))
        if kl > .02:
            break
    return dict(trace=trace, density_errors=errors, Actor_Adam=len(trace), V_Adam=len(trace),
                value_warmup_Adam=0, normalization_mean=mean, normalization_sd=sd,
                equal_episode_weight=1/len(batch))
