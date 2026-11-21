"""One empirical reference-functional natural step, frozen original backbone."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t678_reference_target_pilot'))
import t678_learning as Base
B,M,T,torch,np=Base.B,Base.M,Base.T,Base.torch,Base.np
read,sha,digest=Base.read,Base.sha,Base.digest
optimizer_state,load=Base.optimizer_state,Base.load
data=Base.data

class NaturalStep(torch.optim.Optimizer):
    """Exactly one represented update; scalar root probes do not commit state."""
    def __init__(self,variables):super().__init__(variables,dict(algorithm='T689_reference_functional_natural',lr=1.))
    @torch.no_grad()
    def step(self,displacement):
        assert not self.state;k=0
        for v in self.param_groups[0]['params']:
            v.add_(displacement[k:k+v.numel()].reshape_as(v));k+=v.numel()
            self.state[v]['step']=1
        assert k==11906

def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=M.L.frozen(p,p['initial']['path']).train().requires_grad_(False)
    q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];assert sum(v.numel() for v in variables)==11906
    return q,NaturalStep(variables)

def flat(q):return torch.cat([v.detach().flatten() for v in q.parameters() if v.requires_grad]).clone()
def represent(q,vector):
    k=0
    with torch.no_grad():
        for v in q.parameters():
            if v.requires_grad:v.copy_(vector[k:k+v.numel()].reshape_as(v));k+=v.numel()
    assert k==11906

def metric_data(q,keep,p):
    protected=set(p['before_success_ids']);assert len(protected)==384 and len(set(keep['ids'])&protected)==384
    weight=[]
    for key,n in zip(keep['ids'],keep['lengths']):weight.extend([1/(384*n) if key in protected else 0.]*n)
    with torch.no_grad():initial=M.location(q,keep).detach().clone()
    return dict(f=keep['f'],logits=keep['logits'],initial=initial,weight=initial.new_tensor(weight))

def chunks(metric,p):
    for k in range(0,len(metric['f']),p['metric_chunk']):yield {name:value[k:k+p['metric_chunk']] for name,value in metric.items()}

def kl(q,metric,p):
    value=0.
    with torch.no_grad():
        for batch in chunks(metric,p):
            per=.5*((M.location(q,batch)-batch['initial'])/p['raw_sigma'][0]).square().sum(-1)
            value+=float((per*batch['weight']).sum())
    return value

def hvp(q,metric,vector,p,guard,damped=True):
    variables=[v for v in q.parameters() if v.requires_grad];answer=torch.zeros_like(vector)
    for batch in chunks(metric,p):
        guard();per=.5*((M.location(q,batch)-batch['initial'])/p['raw_sigma'][0]).square().sum(-1)
        first=torch.autograd.grad((per*batch['weight']).sum(),variables,create_graph=True)
        dot=(torch.cat([g.flatten() for g in first])*vector).sum()
        second=torch.autograd.grad(dot,variables);answer+=torch.cat([v.detach().flatten() for v in second])
    if damped:answer+=p['metric_damping']*vector
    return answer

def joint_gradient(p,q,rows,keep,guard):
    q.zero_grad(set_to_none=True);pg=0.;panel={}
    for key,rr in rows.items():
        guard();g,y=B.A.group(q,rr,p);term=Base.Prior.pg(q,g,p)/407;term.backward();pg+=float(term.detach())
        if key in p['gradient_panel_ids']:
            with torch.no_grad():
                for item in g:item['initial']=M.location(q,item).detach().clone()
            panel[key]=g
    loss,losses=M.retention(q,keep,False);(p['replay_weight']*loss).backward()
    gradient=torch.cat([v.grad.detach().flatten() for v in q.parameters() if v.requires_grad]).clone()
    with np.load(p['outcome_gradients'],allow_pickle=False) as saved:expected=-saved['task_gradients'].mean(0)
    with np.load(p['prior_target_gradients'],allow_pickle=False) as saved:expected+=p['replay_weight']*saved['executed_retention_gradient']
    error=float(np.max(np.abs(gradient.numpy()-expected)));assert error<1e-10
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm())<1.,'Original global clip must remain inactive'
    return gradient,panel,dict(PG_loss=pg,keep_initial=float(loss.detach()),reference_worst_initial=float(losses.max().detach()),gradient_error=error,
        full_joint_gradient_norm=float(gradient.norm()),original_norm_clip_inactive=True)

def metric_gradient(q,metric,p,guard):
    variables=[v for v in q.parameters() if v.requires_grad];out=torch.zeros_like(flat(q))
    for batch in chunks(metric,p):
        guard();per=.5*((M.location(q,batch)-batch['initial'])/p['raw_sigma'][0]).square().sum(-1)
        value=torch.autograd.grad((per*batch['weight']).sum(),variables)
        out+=torch.cat([v.detach().flatten() for v in value])
    return out

def ray_value_slope(q,metric,direction,p,guard):
    variables=[v for v in q.parameters() if v.requires_grad];value=0.;slope=0.
    for batch in chunks(metric,p):
        guard();per=.5*((M.location(q,batch)-batch['initial'])/p['raw_sigma'][0]).square().sum(-1)
        loss=(per*batch['weight']).sum();grad=torch.autograd.grad(loss,variables)
        value+=float(loss.detach());slope+=float(torch.cat([v.detach().flatten() for v in grad])@direction)
    return value,slope

def metric_fixture(q,metric,p,guard):
    theta=flat(q);rng=np.random.default_rng(68920261121)
    v=torch.tensor(rng.standard_normal(11906));v/=v.norm()
    w=torch.tensor(rng.standard_normal(11906));w/=w.norm()
    fv=hvp(q,metric,v,p,guard,False);fw=hvp(q,metric,w,p,guard,False)
    try:
        represent(q,theta+p['metric_numeric_epsilon']*v);plus=metric_gradient(q,metric,p,guard)
        represent(q,theta-p['metric_numeric_epsilon']*v);minus=metric_gradient(q,metric,p,guard)
    finally:represent(q,theta)
    numeric=(plus-minus)/(2*p['metric_numeric_epsilon']);error=float((numeric-fv).abs().max())
    symmetry=abs(float(v@fw-w@fv));positive=float(v@fv)
    assert bool(torch.isfinite(fv).all()) and positive>=-1e-10
    assert error<=1e-6+1e-4*float(fv.abs().max()) and symmetry<1e-9
    return dict(max_HVP_central_difference_error=error,bilinear_symmetry_error=symmetry,unit_direction_positive_curvature=positive,
        exact_parameter_restore=torch.equal(flat(q),theta),fixed_seed=68920261121)

def proposal(p,q,rows,keep,guard):
    initial=flat(q);gradient,panel,info=joint_gradient(p,q,rows,keep,guard);metric=metric_data(q,keep,p)
    fixture=metric_fixture(q,metric,p,guard)
    control=M.L.frozen(p,p['control_model']['path']);assert M.L.fixed_hash(control)==M.L.fixed_hash(q)
    target=kl(control,metric,p);assert target>1e-12;del control
    rhs=-gradient;direction=torch.zeros_like(rhs);residual=rhs.clone();search=residual.clone();rr=float(residual@residual);rr0=rr;trace=[]
    for k in range(p['cg_iterations']):
        av=hvp(q,metric,search,p,guard);curvature=float(search@av);assert curvature>0 and np.isfinite(curvature)
        a=rr/curvature;direction+=a*search;residual-=a*av;new=float(residual@residual)
        trace.append(dict(iteration=k,residual_ratio=(new/rr0)**.5,positive_curvature=curvature))
        if new<=rr0*p['cg_early_residual_ratio']**2:rr=new;break
        search=residual+(new/rr)*search;rr=new
    recurrence_residual_ratio=(rr/rr0)**.5
    fd=hvp(q,metric,direction,p,guard,False);curvature=float(direction@fd);assert curvature>0
    actual_residual=rhs-fd-p['metric_damping']*direction;residual_ratio=float(actual_residual.norm()/rhs.norm())
    assert residual_ratio<=p['cg_max_residual_ratio'] and float(gradient@direction)<0
    analytic_scale=(2*target/curvature)**.5;upper=2*analytic_scale;evaluations=0;branch=[];root_trace=[]
    try:
        previous_scale=0.;previous_value=0.;bracket=None
        for j in range(1,p['root_branch_knots']+1):
            scale=upper*j/p['root_branch_knots'];represent(q,initial+scale*direction)
            value,slope=ray_value_slope(q,metric,direction,p,guard);evaluations+=1
            assert np.isfinite(value) and np.isfinite(slope) and slope>0 and value>=previous_value,'Refuse observed scalar branch reversal'
            branch.append(dict(scale=scale,mean_KL=value,directional_slope=slope))
            if bracket is None and previous_value<=target<=value:bracket=(previous_scale,scale,previous_value,value)
            previous_scale,previous_value=scale,value
        assert bracket is not None,'Fixed scalar matching bracket rejected; do not enlarge'
        lo,hi,low,high=bracket
        for k in range(p['root_iterations']):
            guard();scale=(lo+hi)/2;represent(q,initial+scale*direction)
            observed,slope=ray_value_slope(q,metric,direction,p,guard);evaluations+=1
            assert np.isfinite(observed) and np.isfinite(slope) and slope>0 and low<=observed<=high,'Refuse observed scalar root branch reversal'
            root_trace.append(dict(iteration=k,scale=scale,mean_KL=observed,directional_slope=slope,bracket_low=lo,bracket_high=hi))
            if abs(observed-target)<=max(1e-12,p['root_relative_tolerance']*target):break
            if observed<target:lo,low=scale,observed
            else:hi,high=scale,observed
        assert abs(observed-target)<=max(1e-12,p['root_relative_tolerance']*target),'Fixed scalar root did not match'
        displacement=flat(q)-initial
        metrics={key:B.metrics(q,g,p) for key,g in panel.items()}
    finally:represent(q,initial)
    assert torch.equal(flat(q),initial)
    info.update(control_reference384_mean_latent_KL=target,matched_reference384_mean_latent_KL=observed,metric_numeric=fixture,
        root_scale=scale,analytic_quadratic_scale=analytic_scale,scalar_representation_probes=evaluations,
        inspected_scalar_branch=branch,scalar_root_trace=root_trace,positive_root_directional_slope=slope,branch_checks_not_continuum_monotonicity_proof=True,
        cg_trace=trace,cg_residual_ratio=residual_ratio,cg_recurrence_residual_ratio=recurrence_residual_ratio,natural_direction_norm=float(direction.norm()),
        gradient_dot_natural_direction=float(gradient@direction),direction_metric_curvature=curvature,
        direction_damping_fraction=float(p['metric_damping']*(direction@direction)/(curvature+p['metric_damping']*(direction@direction))),
        quadratic_mean_KL_at_root=.5*scale**2*curvature,nonlinear_minus_quadratic_KL=observed-.5*scale**2*curvature,
        proposed_displacement_hash=digest(displacement),proposed_displacement_norm=float(displacement.norm()),
        nonlinear_panel=metrics,max_actual_sigma_task_mean_KL=max(v['task_episode_mean_KL'] for v in metrics.values()),
        metric_is_reference_distribution_latent_Gaussian_not_command_Fisher=True,
        scalar_root_matches_historical_control_not_performance_or_KL_gate=True,
        no_scope_change_or_T687_rescue=True)
    return displacement,info,metric

def fit(p,q,opt,rows,unused,keep,guard,event):
    assert p['Actor_steps']==1 and not opt.state and p['target_arm']=='executed'
    before=optimizer_state(q,opt);fixed=M.L.fixed_hash(q);theta=flat(q)
    delta,info,unused_metric=proposal(p,q,rows,keep,guard)
    event('BEFORE',0);opt.step(delta);event('AFTER',0)
    represented=flat(q)-theta;error=float((represented-delta).abs().max());assert error<1e-12
    committed_KL=kl(q,unused_metric,p)
    assert abs(committed_KL-info['control_reference384_mean_latent_KL'])<=max(1e-12,p['root_relative_tolerance']*info['control_reference384_mean_latent_KL'])
    accepted=bool(torch.isfinite(represented).all()) and info['max_actual_sigma_task_mean_KL']<=p['max_task_latent_KL']
    proposed=optimizer_state(q,opt)
    if not accepted:load(q,opt,before);assert digest(optimizer_state(q,opt))==digest(before)
    assert M.L.fixed_hash(q)==fixed
    info.update(Actor_optimizer=1,optimizer_kind='REFERENCE_FUNCTIONAL_NATURAL_STEP',retained_steps=int(accepted),accepted=accepted,
        represented_update_error=error,committed_reference384_mean_KL=committed_KL,actual_displacement_hash=digest(represented),new_Value=0,
        sampled_surrogate_and_KL_not_closed_loop_guarantee=True)
    return info,proposed
