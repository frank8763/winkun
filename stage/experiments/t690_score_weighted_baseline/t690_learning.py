"""One ordinary Adam step; only the detached peer baseline estimator changes."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t678_reference_target_pilot'))
import t678_learning as Base
B,M,T,torch,np=Base.B,Base.M,Base.T,Base.torch,Base.np
read,sha,digest=Base.read,Base.sha,Base.digest
make,optimizer_state,load=Base.make,Base.optimizer_state,Base.load

def data(p,q,guard):
    rows,unused,keep=Base.data(p,q,guard);seeds=[]
    # Canonical t665 -> t478 wrapper -> t477 wrapper -> original worker.
    # Inspect the actual delegating chain, not just the outer wrapper text.
    outer=T.N.G.T.worker;middle=T.N.G.T.B.worker;worker=T.N.G.T.B.M.worker
    assert 'B.worker(pipe,index,p,counts)' in inspect.getsource(outer)
    assert 'M.worker(pipe,index,p,counts)' in inspect.getsource(middle)
    source=inspect.getsource(worker)
    assert "np.random.default_rng(np.random.SeedSequence(req['seed']))" in source
    for group in rows.values():
        for row in group:
            guard();seed=tuple(row['seed']);assert len(seed)==3 and seed[0]==67220261120
            a=T.R.E.N.S.arrays(row['path']);expected=np.random.default_rng(np.random.SeedSequence(seed)).normal(size=a['innovations'].shape)
            assert np.array_equal(a['innovations'],expected),'Saved episode innovations differ from their independent seed-reset stream'
            seeds.append(seed)
    assert len(seeds)==len(set(seeds))==1724
    stream=dict(unique_episode_seed_count=1724,all_saved_innovation_prefixes_exact=True,seed_array_hash=digest(np.asarray(seeds,dtype=np.int64)),
        worker_file=inspect.getsourcefile(worker),worker_source_hash=digest(source),delegating_workers=[inspect.getsourcefile(outer),inspect.getsourcefile(middle)],
        generator_reset_per_request_verified=True,distinct_PRNG_streams_not_finite_sample_independence_proof=True)
    return rows,stream,keep

def peer_baseline(y,norms):
    y=np.asarray(y,dtype=np.float64);norms=np.asarray(norms,dtype=np.float64)
    assert len(y)>1 and y.shape==norms.shape and np.isfinite(norms).all() and (norms>=0).all()
    baseline=[];fallback=[]
    for i in range(len(y)):
        peers=np.arange(len(y))!=i;total=float(norms[peers].sum())
        baseline.append(float(y[peers]@norms[peers]/total) if total>0 else float(y[peers].mean()))
        fallback.append(total==0.)
    return np.array(baseline),fallback

def fixtures():
    y=np.array([1.,-1.,0.,1.]);w=np.array([2.,3.,5.,7.]);b,_=peer_baseline(y,w)
    ordinary=(y.sum()-y)/3;equal,_=peer_baseline(y,np.ones(4));assert np.max(np.abs(equal-ordinary))<1e-15
    altered_y=y.copy();altered_y[0]=-2.;altered_w=w.copy();altered_w[0]=9000.
    assert peer_baseline(altered_y,altered_w)[0][0]==b[0]
    order=np.array([2,0,3,1]);assert np.max(np.abs(peer_baseline(y[order],w[order])[0]-b[order]))<1e-15
    shifted,_=peer_baseline(y+4,w);assert np.max(np.abs((y+4-shifted)-(y-b)))<1e-15
    zero,fallback=peer_baseline(y,np.zeros(4));assert np.max(np.abs(zero-ordinary))<1e-15 and all(fallback)
    same,_=peer_baseline(np.ones(4),w);assert np.array_equal(same,np.ones(4))
    return dict(equal_norms_are_original_LOO=True,own_return_and_score_excluded=True,permutation_equivariant=True,
        reward_constant_shift_invariant=True,exact_zero_peer_weight_fallback=True,constant_returns_zero_advantage=True)

def weighted_group(q,rows,p):
    group,y=B.A.group(q,rows,p);n=len(y);variables=[v for v in q.parameters() if v.requires_grad]
    ordinary=np.array([g['adv'] for g in group]);same=len(set(y))==1
    if same:
        # Algebraically identical for any nonnegative peer scores, including
        # the predefined zero-denominator fallback; no score norm is measured.
        baseline=np.array(y,dtype=np.float64);fallback=None;norms=None;expected=np.zeros(11906);dispersion=None
    else:
        scores=[]
        for item in group:
            score=M.P.L.density(M.location(q,item),item['raw'],item['atom'],p['raw_sigma']).sum()/p['score_scale']
            scores.append(B.A.flatten_gradient(score,variables))
        scores=np.stack(scores);norms=np.square(scores).sum(1);baseline,fallback=peer_baseline(y,norms)
        advantage=np.asarray(y)-baseline;expected=-(advantage[:,None]*scores).mean(0)
        def spread(a):
            c=a[:,None]*scores
            return float(np.square(c-c.mean(0)).sum(1).mean())
        dispersion=dict(ordinary=spread(ordinary),weighted=spread(advantage),dependent_sample_dispersion_NOT_unbiased_gradient_variance=True)
    for item,value,base in zip(group,np.asarray(y)-baseline,baseline):item.update(adv=float(value),detached_peer_baseline=float(base))
    assert np.isfinite(expected).all() and np.max(np.abs(np.asarray(y)-baseline))<=2.
    detail=dict(outcomes=y,baseline=baseline.tolist(),advantages=[g['adv'] for g in group],ordinary_advantages=ordinary.tolist(),
        score_norms_squared=norms.tolist() if norms is not None else None,score_norms_not_computed_for_constant_returns=same,
        zero_denominator_fallback=fallback,empirical_contribution_dispersion=dispersion,
        baseline_is_detached_initial_peer_only=True,no_signal_is_not_protection=same)
    return group,expected,detail

def numeric(q,group,keep,p):
    variables=[v for v in q.parameters() if v.requires_grad];before=torch.cat([v.detach().flatten() for v in variables]).clone()
    # Baselines in group were frozen at theta0. Recomputing peer norms here
    # would differentiate another objective and leak the control-variate fit.
    def loss():return Base.Prior.pg(q,group,p)+p['replay_weight']*M.retention(q,keep,False)[0]
    gradient=B.A.flatten_gradient(loss(),variables)
    d=np.random.default_rng(69020261121).standard_normal(11906);d/=np.linalg.norm(d)
    def move(vec):
        k=0
        with torch.no_grad():
            for v in variables:v.copy_(vec[k:k+v.numel()].reshape_as(v));k+=v.numel()
    eps=1e-6
    try:
        move(before+torch.tensor(eps*d));plus=float(loss().detach())
        move(before-torch.tensor(eps*d));minus=float(loss().detach())
    finally:move(before)
    actual=(plus-minus)/(2*eps);expected=float(gradient@d);error=abs(actual-expected)
    assert error<=1e-8+1e-4*abs(expected)
    assert torch.equal(torch.cat([v.detach().flatten() for v in variables]),before)
    return dict(analytic=expected,numeric=actual,error=error,baseline_frozen_during_probe=True)

def gradients(p,q,rows,keep,guard):
    variables=[v for v in q.parameters() if v.requires_grad];q.zero_grad(set_to_none=True)
    expected=np.zeros(11906);ordinary=np.zeros(11906);panel={};details={};pg=0.;probe=None
    for key,rr in rows.items():
        guard();group,target,detail=weighted_group(q,rr,p);details[key]=detail
        term=Base.Prior.pg(q,group,p)/407;term.backward();pg+=float(term.detach());expected+=target/407
        if key in p['gradient_panel_ids']:
            with torch.no_grad():
                for item in group:item['initial']=M.location(q,item).detach().clone()
            panel[key]=group
        if key=='dev_task6_00':probe=group
    assert probe is not None
    # Independent directional autograd uses autograd.grad and does not alter
    # the already accumulated parameter .grad buffers.
    numbers=numeric(q,probe,keep,p)
    keep_loss,losses=M.retention(q,keep,False);(p['replay_weight']*keep_loss).backward()
    actual=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
    with np.load(p['prior_target_gradients'],allow_pickle=False) as saved:expected+=p['replay_weight']*saved['executed_retention_gradient']
    with np.load(p['outcome_gradients'],allow_pickle=False) as saved:ordinary=-saved['task_gradients'].mean(0)
    with np.load(p['prior_target_gradients'],allow_pickle=False) as saved:ordinary+=p['replay_weight']*saved['executed_retention_gradient']
    error=float(np.max(np.abs(actual-expected)));assert error<1e-10
    assert np.isfinite(actual).all() and np.linalg.norm(actual)>0
    info=dict(PG_loss_initial=pg,keep_before=float(keep_loss.detach()),reference_worst_before=float(losses.max().detach()),
        gradient_error=error,weighted_gradient_hash=digest(actual),ordinary_stored_gradient_hash=digest(ordinary),
        weighted_vs_ordinary_cosine=float(actual@ordinary/(np.linalg.norm(actual)*np.linalg.norm(ordinary))),
        weighted_vs_ordinary_relative_difference=float(np.linalg.norm(actual-ordinary)/np.linalg.norm(ordinary)),
        baselines=details,numeric=numbers,fixtures=fixtures(),all407_tasks_verified=True,
        expected_first_gradient_target_preserved_not_identical_finite_PPO_objective=True)
    return panel,info

def fit(p,q,opt,rows,unused,keep,guard,event):
    assert p['Actor_steps']==1 and not opt.state and p['target_arm']=='executed'
    before=optimizer_state(q,opt);fixed=M.L.fixed_hash(q);variables=[v for v in q.parameters() if v.requires_grad]
    theta=torch.cat([v.detach().flatten() for v in variables]).clone();panel,info=gradients(p,q,rows,keep,guard)
    norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
    clipped=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
    event('BEFORE',0);opt.step();event('AFTER',0)
    delta=(torch.cat([v.detach().flatten() for v in variables])-theta).numpy().copy()
    assert q.phash()!=p['control_policy_hash'],'Do not repeat an identical already consumed candidate'
    formula=-p['fit_lr']*clipped/(np.abs(clipped)+p['Adam_eps']);error=float(np.max(np.abs(delta-formula)));assert error<1e-12
    values={key:B.metrics(q,g,p) for key,g in panel.items()};maxkl=max(v['task_episode_mean_KL'] for v in values.values())
    finite=all(bool(torch.isfinite(v).all()) for v in variables);accepted=finite and maxkl<=p['max_task_latent_KL'];proposed=optimizer_state(q,opt)
    if not accepted:load(q,opt,before);assert digest(optimizer_state(q,opt))==digest(before)
    assert M.L.fixed_hash(q)==fixed
    info.update(Actor_Adam=1,retained_steps=int(accepted),target_arm='executed',baseline_arm='SCORE_NORM_WEIGHTED_PEER_LOO',
        gradient_norm_before_clip=norm,first_Adam_formula_error=error,actual_displacement_hash=digest(delta),actual_displacement_norm=float(np.linalg.norm(delta)),
        max_actual_sigma_task_mean_KL=maxkl,finite=finite,accepted=accepted,nonlinear_panel=values,
        old_surrogate_diagnostics_not_safety_acceptance=True,no_Value_or_new_reward=True,innovation_streams=unused)
    return info,proposed
