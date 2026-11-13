"""Finite Adam proposals accepted only within stored-output hard constraints.

Backtracking scales actual post-Adam parameter displacement. Accepted moments
remain the proposal moments; rejected proposal restores BOTH parameters/Adam.
This is not TRPO, constrained RL or a guarantee on unvisited states.
"""
from pathlib import Path
import copy,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t516_nonlinear_compatibility'))
import t516_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest
Actor,variables,fixed_hash,make,frozen,export,data,location,loss,metrics=B.Actor,B.variables,B.fixed_hash,B.make,B.frozen,B.export,B.data,B.location,B.loss,B.metrics

def keep_limits(q,rows,p):
    maxrmse,maxabs=0.,0.
    with torch.no_grad():
        for g in rows:
            if g['kind']=='repair':continue
            d=location(q,g)-g['target']
            maxrmse=max(maxrmse,float(d.square().mean(0).sqrt().max()))
            maxabs=max(maxabs,float(d.abs().max()))
    return dict(max_RMSE=maxrmse,max_abs=maxabs,feasible=bool(maxrmse<=p['keep_RMSE_limit'] and maxabs<=p['keep_max_abs_limit']))

def choose(q,params,before,proposed,rows,p,repair_before):
    trials=[];accepted=None
    for alpha in p['backtracking_factors']:
        with torch.no_grad():
            for z,a,b in zip(params,before,proposed):z.copy_(a+alpha*(b-a))
        limits=keep_limits(q,rows,p)
        with torch.no_grad():repair=float(loss(q,next(g for g in rows if g['kind']=='repair'),p))
        ok=limits['feasible'] and math.isfinite(repair) and repair<repair_before-p['minimum_repair_decrease']
        trials.append([alpha,limits['max_RMSE'],limits['max_abs'],repair,bool(ok)])
        if ok:accepted=alpha;break
    return accepted,trials

def tangent_projection(q,rows,p,displacement):
    params=variables(q);vectors=[];labels=[]
    for group_index,g in enumerate(rows):
        if g['kind']=='repair':continue
        d=location(q,g)-g['target'];mse=d.square().mean(0);absflat=d.abs().flatten();maximum,index=absflat.max(0)
        scalars=[]
        for axis in range(2):
            if float(mse[axis].detach().sqrt())>=p['active_constraint_fraction']*p['keep_RMSE_limit']:
                scalars.append((mse[axis],3*group_index+axis))
        if float(maximum.detach())>=p['active_constraint_fraction']*p['keep_max_abs_limit']:
            scalars.append((absflat[index],3*group_index+2))
        for scalar,label in scalars:
            gradients=torch.autograd.grad(scalar,params,retain_graph=True)
            v=torch.cat([z.detach().reshape(-1) for z in gradients]);norm=float(v.norm())
            if norm>1e-15:vectors.append(v/norm);labels.append(label)
    if not vectors:return displacement,dict(active_constraints=[],rank=0,retained_direction_ratio=1.,linearized_residual=0.)
    j=torch.stack(vectors);u,s,vh=torch.linalg.svd(j,full_matrices=False)
    rank=int((s>p['projection_svd_relative_tolerance']*s[0]).sum())
    basis=vh[:rank];projected=displacement-basis.T@(basis@displacement)
    assert torch.isfinite(projected).all()
    residual=float((j@projected).norm());denom=max(float((j@displacement).norm()),1e-30)
    return projected,dict(active_constraints=labels,rank=rank,retained_direction_ratio=float(projected.norm())/max(float(displacement.norm()),1e-30),linearized_residual=residual,linearized_relative_residual=residual/denom)


def fit(obj,rows,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict())
    initial=metrics(q,rows,p);assert keep_limits(q,rows,p)['feasible']
    trace=[];accepted_count=0;trial_count=0;reason='PROPOSAL_LIMIT_COMPLETE'
    params=variables(q);initial_counter=[int(ao.state[z]['step']) if z in ao.state else 0 for z in params];repair_group=next(g for g in rows if g['kind']=='repair')
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();before=[z.detach().clone() for z in params];before_opt=copy.deepcopy(ao.state_dict())
        with torch.no_grad():repair_before=float(loss(q,repair_group,p))
        ao.zero_grad(set_to_none=True);total=dict(repair=0.,keep=0.,prefix=0.)
        for g in rows:
            if g['kind']=='repair':weight=.5
            elif g['kind']=='prefix':weight=.1
            else:weight=.4/(2*sum(h['kind']=='keep' and h['family']==g['family'] for h in rows))
            z=loss(q,g,p);(weight*z).backward();total[g['kind']]+=weight*float(z.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();event('AFTER',k)
        proposed=[z.detach().clone() for z in params]
        displacement=torch.cat([(b-a).reshape(-1) for a,b in zip(before,proposed)])
        with torch.no_grad():
            for z,a in zip(params,before):z.copy_(a)
        projected,projection=tangent_projection(q,rows,p,displacement)
        proposed=[];offset=0
        for a in before:
            proposed.append(a+projected[offset:offset+a.numel()].reshape_as(a));offset+=a.numel()
        assert offset==len(projected)
        alpha,trials=choose(q,params,before,proposed,rows,p,repair_before)
        if alpha is None:
            with torch.no_grad():
                for z,a in zip(params,before):z.copy_(a)
            ao.load_state_dict(before_opt)
            assert all(torch.equal(z,a) for z,a in zip(params,before)) and digest(ao.state_dict())==digest(before_opt)
            reason='NO_REGISTERED_PROJECTED_FEASIBLE_DESCENT_STEP'
        else:accepted_count+=1
        entry=dict(proposal=k,alpha=alpha,trials=trials,gradient_norm=norm,repair_before=repair_before,weighted_losses=total,
                   accepted=alpha is not None,rollback_exact=alpha is None,projection=projection)
        trace.append(entry);event('DECISION',entry);trial_count+=len(trials)
        assert fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh and keep_limits(q,rows,p)['feasible']
        if alpha is None:break
    assert all(int(ao.state[z]['step'])==n+accepted_count for z,n in zip(params,initial_counter)) if ao.state else accepted_count==0
    return dict(Actor_Adam=len(trace),accepted_parameter_updates=accepted_count,rejected_parameter_updates=len(trace)-accepted_count,
                line_search_function_trials=trial_count,stop_reason=reason,trace_sha256=digest(trace),Value_Adam=0,new_policy_gradient_updates=0,
                initial=initial,final=metrics(q,rows,p),trainable_parameters=13986,frozen_parent_hash=fixed,
                final_keep_constraints=keep_limits(q,rows,p),post_Adam_displacement_not_raw_gradient=True,active_tangent_projection=True)
