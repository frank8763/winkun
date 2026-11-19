"""Mean Adam versus actual-displacement old-outcome constraint, same data.

Two sampled old task surrogate losses are local objectives, not safety bounds.
"""
from pathlib import Path
import copy,itertools,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t652_parameter_scope_diagnosis'))
import t652_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
make,state,load,export=Base.make,Base.state,Base.load,Base.export
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
B=Base.Prior.Prior
outcome=Base.Prior.outcome


def build(q,groups,p):
    result=[];diag=[];device=next(q.parameters()).device
    for task,group in groups:
        n=len(group);assert n in [4,16];rewards=[outcome(row) for a,row in group];parts=[]
        for i,(a,row) in enumerate(group):
            assert a['stochastic'].all() and not a['ownership'].any()
            x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
            with torch.no_grad():
                loc=q.sequence(x);assert float((loc-torch.tensor(a['locations'],device=device)).abs().max())<1e-10
                assert float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())<1e-9
            baseline=sum(rewards[j] for j in range(n) if j!=i)/(n-1)
            parts.append(dict(id=task,x=x,raw=raw,atom=atom,old=old,initial=loc.detach(),adv=rewards[i]-baseline,source_sha256=row['sha256']))
        result.append(parts);diag.append(dict(id=task,n=n,outcomes=rewards,advantages=[v['adv'] for v in parts],informative=len(set(rewards))>1))
    assert len(result)==21
    return result,diag


def task_loss(q,group,p):
    total=next(q.parameters()).new_zeros(())
    for b in group:
        loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        total-=torch.minimum(ratio*b['adv'],ratio.clamp(1-p['ratio_clip'],1+p['ratio_clip'])*b['adv']).sum()/(len(group)*p['score_scale'])
    return total


def variables(q):return [v for v in q.parameters() if v.requires_grad]
def flat(vv):return torch.cat([v.reshape(-1) for v in vv])


def project(delta,gradients,p):
    """Euclidean projection onto two loss halfspaces, active-set enumeration."""
    norms=torch.linalg.vector_norm(gradients,dim=1);assert bool((norms>p['constraint_gradient_floor']).all()),'Old outcome constraint gradient uninformative'
    g=gradients/norms[:,None];n=len(g);assert n==2
    tol=p['projection_tolerance'];candidates=[]
    for count in range(n+1):
        for active in itertools.combinations(range(n),count):
            if active:
                a=g[list(active)];gram=a@a.T;lam=torch.linalg.pinv(gram,rtol=1e-12)@(a@delta);candidate=delta-a.T@lam
                if bool((lam<-tol).any()):continue
            else:candidate=delta
            if bool((g@candidate<=tol).all()):candidates.append((float((candidate-delta).square().sum()),active,candidate))
    assert candidates,'Two-halfspace projection found no feasible candidate'
    _,active,value=min(candidates,key=lambda v:v[0])
    return value,dict(active_constraints=list(active),initial_unit_dots=(g@delta).tolist(),projected_unit_dots=(g@value).tolist(),
        old_gradient_norms=norms.tolist(),Adam_delta_L2=float(delta.norm()),projected_L2=float(value.norm()),
        projection_distance=float((value-delta).norm()),finite_halfspace_only=True)


def set_displacement(obj,before,proposal,displacement,scale):
    load(obj,proposal)
    if scale==0:load(obj,before);return
    offset=0
    with torch.no_grad():
        for name,v in obj[0].named_parameters():
            if v.requires_grad:
                n=v.numel();v.copy_(before['actor'][name]+scale*displacement[offset:offset+n].reshape_as(v));offset+=n
    assert offset==len(displacement)


@torch.no_grad()
def old_values(q,items,p):return [float(task_loss(q,g,p)) for g in items]


@torch.no_grad()
def trust(q,items,p):
    return [sum(float((.5*((q.sequence(b['x'])-b['initial'])/.2).square().sum(-1)).mean()) for b in group)/len(group) for group in items]


def fit(obj,old,groups,p,guard,event):
    q,opt=obj;body=L.fixed_hash(q);items,diag=build(q,groups,p)
    by={g[0]['id']:g for g in items};protected=[by[key] for key in p['protected_tasks']]
    assert all(len(set(v['outcomes']))>1 for v in diag if v['id'] in p['protected_tasks'])
    params=variables(q);trace=[]
    for k in range(p['fit_steps']):
        guard();before=state(obj);opt.zero_grad(set_to_none=True)
        old_grad=[];old_before=[]
        for group in protected:
            loss=task_loss(q,group,p);gg=torch.autograd.grad(loss,params)
            old_grad.append(flat([v.detach() for v in gg]));old_before.append(float(loss.detach()))
        gradients=torch.stack(old_grad);opt.zero_grad(set_to_none=True);learning=0.
        for group in items:
            loss=task_loss(q,group,p)/len(items);loss.backward();learning+=float(loss.detach())
        keep=B.keep_backward(q,old,p);norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj)
        delta=flat([(v-before['actor'][name]).detach() for name,v in q.named_parameters() if v.requires_grad])
        informative=bool((torch.linalg.vector_norm(gradients,dim=1)>p['constraint_gradient_floor']).all())
        if informative:projected,detail=project(delta,gradients,p)
        else:
            projected=torch.zeros_like(delta)
            detail=dict(reason='At least one sampled old-task gradient vanished',constraint_informative=False,
                old_gradient_norms=torch.linalg.vector_norm(gradients,dim=1).tolist())
        trials=[];accepted=False;chosen=0.;old_after=None;kl=None
        scales=[1.] if p['arm']=='mean' else p['constraint_scales']
        for scale in scales:
            if p['arm']=='mean':load(obj,proposal)
            else:set_displacement(obj,before,proposal,projected,scale)
            old_after=old_values(q,protected,p);kl=trust(q,items,p)
            finite=bool(np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL'])
            no_old_increase=all(a<=b+p['old_surrogate_tolerance'] for a,b in zip(old_after,old_before))
            moved=float((delta if p['arm']=='mean' else projected).norm())*scale>p['minimum_displacement']
            applied=flat([(v-before['actor'][name]).detach() for name,v in q.named_parameters() if v.requires_grad])
            unit_dots=(gradients@applied)/torch.linalg.vector_norm(gradients,dim=1).clamp_min(p['constraint_gradient_floor'])
            halfspaces=informative and bool((unit_dots<=p['projection_tolerance']).all())
            okay=finite and moved and (p['arm']=='mean' or (no_old_increase and halfspaces))
            trials.append(dict(scale=scale,old_values=old_after,old_nonincrease=no_old_increase,max_task_latent_KL=max(kl),
                actual_unit_gradient_dots=unit_dots.tolist(),actual_halfspaces_pass=halfspaces,moved=moved,accepted=okay))
            if okay:accepted=True;chosen=scale;break
        if not accepted:load(obj,before)
        actual_delta=flat([(v-before['actor'][name]).detach() for name,v in q.named_parameters() if v.requires_grad])
        trace.append(dict(step=k,learning_loss=learning,keep=keep,gradient_norm=norm,old_before=old_before,
            old_after=old_after,projection=detail,actual_old_first_order_changes=(gradients@actual_delta).tolist(),
            proposed_Adam_hash=digest(proposal['actor_opt']),retained_Adam_hash=digest(opt.state_dict()),
            chosen_scale=chosen,accepted=accepted,trials=trials))
        assert L.fixed_hash(q)==body and all(torch.isfinite(v).all() for v in params)
        if not accepted:assert digest(state(obj))==digest(before);break
    return dict(Actor_Adam=len(trace),accepted=sum(v['accepted'] for v in trace),arm=p['arm'],trace=trace,diagnostics=diag,
        trainable_parameters=11906,encoder_GRU_head_frozen_exact=True,full384_each_update=True,
        constraint='actual Adam displacement on two old sample outcome losses plus finite checks',
        partial_commit_moments='full proposed Adam moments; full rollback if rejected',no_closedloop_guarantee=True)
