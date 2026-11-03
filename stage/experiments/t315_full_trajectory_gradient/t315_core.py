"""Full-trajectory gradient, matched controls, no native training."""
from pathlib import Path
import sys,copy,io,gzip,json,time,math,os,shutil,hashlib,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t312_remaining_recovery'))
import recover as K
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t315_full_trajectory_gradient'
DOC=ROOT/'stage/docs/distance_20261103'
read=lambda p:json.loads(Path(p).read_text())

def flat(q):
    return torch.cat([v.detach().reshape(-1) for v in q.parameters() if v.requires_grad]).double()

def put(q,x):
    pos=0
    with torch.no_grad():
        for v in q.parameters():
            if v.requires_grad:
                n=v.numel();v.copy_(x[pos:pos+n].reshape_as(v).to(v.dtype));pos+=n
    assert pos==len(x)

def cg(mul,g,steps,tolerance):
    x=torch.zeros_like(g);r=g.clone();d=r.clone();rr=torch.dot(r,r);initial=float(rr);history=[]
    assert initial>0
    for k in range(steps):
        z=mul(d);curv=torch.dot(d,z);assert torch.isfinite(curv) and float(curv)>0
        alpha=rr/curv;x=x+alpha*d;r=r-alpha*z;new=torch.dot(r,r)
        history.append(dict(iteration=k,residual_squared=float(new),relative_residual_squared=float(new)/initial))
        if float(new)<=initial*tolerance:break
        d=r+(new/rr)*d;rr=new
    assert torch.isfinite(x).all() and float(torch.dot(x,g))>0
    return x,history

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def baseline_batch(p):
    items,values,fitted,screens=K.cached_first_batch(p)
    assert all(screens[f'{a}_0']['value']['heldout_value_gate'] and screens[f'{a}_0']['variance_gate'] for a in ['mc','gae'])
    # Both original arms collected exactly the same first16 histories.
    assert all(a['row']['array_hashes']==b['row']['array_hashes'] and a['row']['integration_sha256']==b['row']['integration_sha256'] for a,b in zip(items['mc'],items['gae']))
    return items,fitted

def fisher_rows(p,q,items,indices,guard):
    params=[v for v in q.parameters() if v.requires_grad];rows=[];selection=[]
    for item,ix in zip(items,indices):
        for k in ix[:p['fisher_rows_per_episode']]:
            guard();a=item['arrays'];index=np.asarray([int(k)])
            value=K.lp(q,a,index).sum();assert float(value.detach())==float(a['logp'][k])
            gg=torch.autograd.grad(value,params,allow_unused=True)
            row=torch.cat([torch.zeros_like(v).reshape(-1) if g is None else g.detach().reshape(-1) for v,g in zip(params,gg)]).double()
            assert torch.isfinite(row).all();rows.append(row);selection.append(dict(id=item['row']['id'],index=int(k)))
    return torch.stack(rows),selection

@torch.no_grad()
def measure(p,q,items,target,arm,ax,ay,offset):
    ds=[];surrogates=[];by={};scale=max(.05,float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in items]))))
    for i,t in zip(items,target):
        a=i['arrays'];delta=K.lp(q,a)-torch.tensor(a['logp']);ds.append(delta)
        adv=torch.tensor(t[arm]*t['prefix']/scale);ratio=delta.exp()
        surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        surrogates.append(surrogate);by.setdefault(i['row']['goal_family'],[]).append(float(surrogate))
    delta=torch.cat(ds);k3=float((delta.exp()-1-delta).mean());mx=float(delta.abs().max())
    retention=torch.stack([(q.step(ax[offset[k]:offset[k+1]])[0]-ay[offset[k]:offset[k+1]]).square().mean() for k in range(8)]).mean()*p['retention_weight']
    obj=float(torch.stack(surrogates).mean()-retention)
    vals=dict(k3=k3,max_abs_logratio=mx,objective=obj,weighted_retention=float(retention),per_goal_surrogate={k:float(np.mean(v)) for k,v in by.items()})
    assert all(math.isfinite(v) for v in [k3,mx,obj,float(retention),*vals['per_goal_surrogate'].values()])
    return vals

def full_gradient(p,q,items,target,arm,ax,ay,old_indices,guard):
    """Only the NEW-task gradient switches to complete equal-weight episodes.

    At the original policy ratio=1 and the clip is inactive. Stream the exact
    first-order log-probability identity; keep the original128 retention rows.
    """
    params=[v for v in q.parameters() if v.requires_grad]
    gradient=torch.zeros_like(flat(q));calls=0;density=[]
    scale=max(.05,float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in items]))))
    def backward(value):
        nonlocal calls
        gg=torch.autograd.grad(value,params,allow_unused=True);calls+=1
        return torch.cat([torch.zeros_like(v).reshape(-1) if g is None else g.detach().reshape(-1) for v,g in zip(params,gg)]).double()
    for i,t in zip(items,target):
        guard();a=i['arrays'];value=K.lp(q,a);error=float((value.detach()-torch.tensor(a['logp'])).abs().max());assert error==0
        coefficient=torch.tensor(t[arm]*t['prefix']/scale)
        gradient+=backward(torch.dot(value,coefficient)/len(coefficient))/16
        density.append(dict(id=i['row']['id'],logp_error=error,rows=len(coefficient)))
    guard();keep=(q.step(ax[old_indices])[0]-ay[old_indices]).square().mean()*p['retention_weight']
    old_gradient=backward(keep);gradient-=old_gradient
    assert calls==17 and torch.isfinite(gradient).all() and all(v.grad is None for v in params)
    return gradient,dict(autograd_calls=calls,new_trajectory_rows=sum(v['rows'] for v in density),old_retention_sample_rows=len(old_indices),
        raw_full_gradient_sha256=K.S.digest(gradient),raw_full_gradient_norm=float(gradient.norm()),old_retention_gradient_norm=float(old_gradient.norm()),density=density)

def science(p,guard,emit,save_model):
    items,fitted=baseline_batch(p);q=K.policy(p);initial=copy.deepcopy(q.state_dict());x0=flat(q)
    parent=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=K.T.anchors(p,parent)
    reference=read(ROOT/'stage/runs/t313_fisher_feasibility/metrics.json');results=[];new_updates=[]
    rng=np.random.default_rng(p['seed']);indices=[rng.integers(0,len(i['returns']),16) for i in items['mc']]
    old_indices=np.concatenate([rng.integers(offset[k],offset[k+1],16) for k in range(8)])
    # SAME original score samples and old retention samples; only g changes.
    J,selection=fisher_rows(p,q,items['mc'],indices,guard);assert len(J)==16*p['fisher_rows_per_episode']
    Jh=K.S.digest(J);count=len(J);assert Jh==reference['score_matrix_sha256'] and selection==reference['fisher_selection']
    def undamped(v):return J.T@(J@v)/count
    def metric(v):return undamped(v)+p['fisher_damping']*v
    emit('fisher_built',dict(rows=count,distinct_id_indices=len({(x['id'],x['index']) for x in selection}),parameters=len(x0),score_matrix_sha256=Jh))
    for arm in ['mc','gae']:
        guard();q.load_state_dict(initial);opt=K.T.L.optimizer(q,p)
        batch,target=fitted[arm]
        raw_gradient,gradient_report=full_gradient(p,q,batch,target,arm,ax,ay,old_indices,guard)
        opt.zero_grad();pos=0;params=[v for v in q.parameters() if v.requires_grad]
        for v in params:
            n=v.numel();v.grad=-raw_gradient[pos:pos+n].reshape_as(v).to(v.dtype).clone();pos+=n
        assert pos==len(raw_gradient)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        g=-torch.cat([v.grad.detach().reshape(-1) for v in params]).double()
        opt.step();opt.zero_grad(set_to_none=True);emit('new_full_gradient_Adam_complete',dict(arm=arm,completed_calls=1))
        raw=measure(p,q,batch,target,arm,ax,ay,offset);adam=flat(q)-x0
        new_updates.append(dict(arm=arm,gradient=gradient_report,preclip_gradient_norm=norm,clipped_uphill_gradient_sha256=K.S.digest(g),
            raw_new_Adam_policy_hash=K.S.digest(q.state_dict()),raw_new_Adam_optimizer_hash=K.S.digest(opt.state_dict()),raw_measurement=raw,
            raw_Adam_state_preserved_by_exact_recipe_and_input_SHA=True))
        q.load_state_dict(initial);base=measure(p,q,batch,target,arm,ax,ay,offset)
        natural,cgh=cg(metric,g,p['cg_steps'],p['cg_relative_squared_tolerance'])
        six=[f for f in base['per_goal_surrogate'] if '_d6_' in f];assert len(six)==2
        for name,direction in [('adam',adam),('empirical_fisher',natural)]:
            guard();assert float(torch.dot(g,direction))>0
            curvature=float(torch.dot(direction,metric(direction)));assert curvature>0 and math.isfinite(curvature)
            alpha=math.sqrt(2*p['max_conditional_k3']/curvature);trials=[];accepted=False
            for k in range(p['backtrack_steps']):
                guard();factor=p['backtrack_decay']**k;put(q,x0+alpha*factor*direction)
                trial=dict(trial=k,factor=factor,initial_metric_scale=alpha,policy_hash=K.S.digest(q.state_dict()),predicted_damped_score_quadratic=.5*(alpha*factor)**2*curvature)
                try:
                    value=measure(p,q,batch,target,arm,ax,ay,offset)
                    improvement=value['objective']-base['objective'];six_changes={f:value['per_goal_surrogate'][f]-base['per_goal_surrogate'][f] for f in six}
                    numeric=value['k3']<=p['max_conditional_k3'] and value['max_abs_logratio']<=p['max_abs_log_ratio']
                    accepted=bool(numeric and improvement>p['objective_min_improvement'] and all(v>=-p['goal_surrogate_tolerance'] for v in six_changes.values()))
                    trial.update(value=value,improvement=improvement,six_goal_surrogate_changes=six_changes,numeric_pass=bool(numeric),accepted=accepted)
                except (AssertionError,RuntimeError,OverflowError,ValueError) as error:
                    trial.update(accepted=False,numeric_pass=False,evaluation_error=repr(error))
                trials.append(trial);emit('offline_parameter_proposal',dict(arm=arm,method=name,**trial))
                if accepted:break
            model=save_model(q,arm,name,accepted)
            results.append(dict(arm=arm,method=name,accepted=accepted,initial_policy_hash=p['policy_hash'],base=base,raw_new_full_gradient_Adam_measurement=raw,
                uphill_gradient_sha256=K.S.digest(g),gradient_source='complete16episodes_old128retention_unchanged',direction_sha256=K.S.digest(direction),direction_norm=float(direction.norm()),direction_dot_gradient=float(torch.dot(direction,g)),
                damped_curvature=curvature,cg_history=cgh if name=='empirical_fisher' else [],trials=trials,model=model))
            q.load_state_dict(initial);assert K.S.digest(q.state_dict())==p['policy_hash']
    return dict(results=results,new_full_gradient_Adam_updates=new_updates,new_full_gradient_Adam_calls=2,historical_Adam_calls=0,new_V_Adam_calls=0,new_native_steps=0,
        score_matrix_sha256=Jh,fisher_selection=selection,fisher_unique_selections=len({(x['id'],x['index']) for x in selection}),fisher_rows=count,
        total_parameter_proposals=sum(len(r['trials']) for r in results),total_candidate_parameter_updates=2+sum(len(r['trials']) for r in results),empirical_fisher_viable_both_arms=all(r['accepted'] for r in results if r['method']=='empirical_fisher'),
        Adam_viable_both_arms=all(r['accepted'] for r in results if r['method']=='adam'),six_metre_navigation_evaluated=False,standard_method_not_novelty_claim=True)
