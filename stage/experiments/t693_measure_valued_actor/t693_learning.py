"""One direct mean-credit Actor step, no learned value and no line search."""
from pathlib import Path
import sys
import t693_estimator as Est
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t690_score_weighted_baseline'))
import t690_learning as Base
B, M, T, torch, np = Base.B, Base.M, Base.T, Base.torch, Base.np
read, sha, digest = Base.read, Base.sha, Base.digest
make, optimizer_state, load = Base.make, Base.optimizer_state, Base.load


def data(p, q, guard):
    folder = Path(p['collection_out'])
    assert read(folder/'audit.json')['state'] == 'PASS'
    assert sha(folder/'CREDITS.json') == p['collection_credit_sha256']
    credits = read(folder/'CREDITS.json'); assert len(credits['records']) == 32
    sources = []
    for entry, expected in zip(credits['records'], p['fixed_records']):
        guard(); assert entry['source_sha256'] == expected['source_sha256']
        assert entry['task'] == expected['task'] and entry['rep'] == expected['rep']
        assert entry['slot'] == expected['slot'] and entry['active'] == expected['active']
        if not entry['active']:
            assert all(x == [0., 0.] for x in entry['credit'].values()); continue
        assert sha(entry['source']) == entry['source_sha256']
        a = T.R.E.N.S.arrays(entry['source']); k = entry['slot']
        g = M.features(q, dict(states=a['states'][:k+1], actions=a['actions'][:k+1]), entry['task'], entry['source_sha256'])
        with torch.no_grad():
            mu = M.location(q, g)[k].numpy()
        assert float(np.abs(mu-np.asarray(expected['mu'])).max()) < 1e-10
        coeff = np.asarray(entry['credit'][p['arm']], np.float64)
        assert coeff.shape == (2,) and np.isfinite(coeff).all()
        sources.append(dict(g=g, slot=k, coefficient=coeff, task=entry['task'], rep=entry['rep']))
    keep = M.pack(M.references(p, q, guard))
    groups = B.dataset(p); panel = {}
    for key in p['gradient_panel_ids']:
        guard(); group, _ = B.A.group(q, groups[key], p)
        with torch.no_grad():
            for g in group: g['initial'] = M.location(q, g).detach().clone()
        panel[key] = group
    assert len(panel) == 68
    return sources, panel, keep


def rl_loss(q, sources):
    loss = next(q.parameters()).new_zeros(())
    for item in sources:
        mu = M.location(q, item['g'])[item['slot']]
        # Mean over all32 histories including masked zero entries. Coefficient
        # has no derivative through suffix return, sampled action or history.
        loss -= (mu*mu.new_tensor(item['coefficient'])).sum()*(Est.SLOTS/Est.SCALE)/32
    return loss


def assign(variables, vector):
    offset = 0
    with torch.no_grad():
        for v in variables:
            v.copy_(vector[offset:offset+v.numel()].reshape_as(v)); offset += v.numel()
    assert offset == len(vector)


def direction(p, q, opt, sources, panel, keep, guard):
    before = optimizer_state(q, opt); assert not opt.state
    variables = [v for v in q.parameters() if v.requires_grad]
    theta = torch.cat([v.detach().flatten() for v in variables]).clone()
    assert len(theta) == 11906
    rl = rl_loss(q, sources)
    raw = torch.cat([v.flatten() for v in torch.autograd.grad(rl, variables)]).detach()
    magnitude = float(raw.norm()); finite = bool(torch.isfinite(raw).all())
    if not finite or magnitude == 0:
        return dict(accepted=False, reason='ZERO_OR_NONFINITE_RL_GRADIENT', RL_gradient_norm=magnitude,
                    RL_gradient_hash=digest(raw), Actor_Adam=0), None
    guard(); opt.zero_grad(set_to_none=True)
    rloss = rl_loss(q, sources); keep_loss, losses = M.retention(q, keep, False)
    loss = rloss+p['replay_weight']*keep_loss
    loss.backward()
    gradient = torch.cat([v.grad.detach().flatten() for v in variables]).clone()
    with np.load(p['prior_target_gradients'], allow_pickle=False) as stored:
        expected = raw.numpy()+p['replay_weight']*stored['executed_retention_gradient']
    error = float(np.abs(gradient.numpy()-expected).max()); assert error < 1e-10
    # Directional difference of the exact frozen-credit objective, not a
    # simulator finite difference and not differentiation through Q or Y.
    d = torch.tensor(np.random.default_rng([Est.SEED, 5]).normal(size=11906)); d /= d.norm()
    def objective(): return rl_loss(q, sources)+p['replay_weight']*M.retention(q, keep, False)[0]
    try:
        assign(variables, theta+1e-6*d); plus = float(objective().detach())
        assign(variables, theta-1e-6*d); minus = float(objective().detach())
    finally: assign(variables, theta)
    analytic, numeric = float(gradient@d), (plus-minus)/2e-6
    numerical_error = abs(analytic-numeric); assert numerical_error < 1e-8+1e-4*abs(analytic)
    norm = float(torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True))
    clipped = torch.cat([v.grad.detach().flatten() for v in variables]).clone()
    formula = -p['fit_lr']*clipped/(clipped.abs()+p['Adam_eps'])
    try:
        assign(variables, theta+formula)
        metrics = {key: B.metrics(q, group, p) for key, group in panel.items()}
        maximum = max(x['task_episode_mean_KL'] for x in metrics.values())
        finite_proposal = all(bool(torch.isfinite(v).all()) for v in variables)
        formula_hash = q.phash()
    finally: load(q, opt, before)
    assert digest(optimizer_state(q, opt)) == digest(before) and not opt.state
    accepted = finite_proposal and maximum <= p['max_task_latent_KL']
    assert formula_hash not in p['consumed_candidate_hashes'], 'Do not repeat an already consumed candidate'
    return dict(accepted=accepted, reason='FORMULA_NUMERICALLY_ADMITTED' if accepted else 'FIXED_FORMULA_KL_OR_FINITE_REJECTED',
                RL_gradient_norm=magnitude, RL_gradient_hash=digest(raw), gradient_hash=digest(gradient),
                gradient_error=error, gradient_norm_before_clip=norm,
                numeric=dict(analytic=analytic, numeric=numeric, error=numerical_error),
                formula_displacement_hash=digest(formula), formula_displacement_norm=float(formula.norm()),
                formula_policy_hash_NOT_actual_export=formula_hash,
                max_actual_sigma_task_mean_KL=maximum, nonlinear_panel=metrics,
                keep_before=float(keep_loss.detach()), worst_reference_before=float(losses.max().detach()),
                actual_optimizer_calls=0, Actor_Adam=0, local8_ID_not_full407_PG=True,
                no_variance_or_retention_guarantee=True), (gradient, formula)


def fit(p, q, opt, sources, panel, keep, guard, event):
    initial = optimizer_state(q, opt); fixed = M.L.fixed_hash(q)
    variables = [v for v in q.parameters() if v.requires_grad]
    theta = torch.cat([v.detach().flatten() for v in variables]).clone()
    info, arrays = direction(p, q, opt, sources, panel, keep, guard)
    if not info['accepted']:
        assert digest(initial) == digest(optimizer_state(q, opt))
        return info, initial
    gradient, formula = arrays
    opt.zero_grad(set_to_none=True); offset = 0
    for v in variables:
        v.grad = gradient[offset:offset+v.numel()].reshape_as(v).clone(); offset += v.numel()
    torch.nn.utils.clip_grad_norm_(variables, 1., error_if_nonfinite=True)
    event('BEFORE', 0); opt.step(); event('AFTER', 0)
    delta = torch.cat([v.detach().flatten() for v in variables])-theta
    error = float((delta-formula).abs().max()); assert error < 1e-12
    assert q.phash() not in p['consumed_candidate_hashes']
    assert M.L.fixed_hash(q) == fixed
    actual_metrics = {key: B.metrics(q, group, p) for key, group in panel.items()}
    maximum = max(x['task_episode_mean_KL'] for x in actual_metrics.values())
    accepted = maximum <= p['max_task_latent_KL'] and all(bool(torch.isfinite(v).all()) for v in variables)
    proposed = optimizer_state(q, opt)
    info.update(accepted=accepted, Actor_Adam=1, actual_optimizer_calls=1,
                actual_first_Adam_formula_error=error, actual_displacement_hash=digest(delta),
                actual_policy_hash=q.phash(), actual_max_task_mean_KL=maximum,
                reason='ACTUAL_STEP_NUMERICALLY_ADMITTED' if accepted else 'ACTUAL_KL_OR_FINITE_REJECTED')
    if not accepted: load(q, opt, initial)
    return info, proposed
