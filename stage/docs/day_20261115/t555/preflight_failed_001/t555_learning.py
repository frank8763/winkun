"""GEM-inspired post-Adam halfspace pilot and local displacement-length control.

Not GEM/A-GEM reproduction, no per-task or closed-loop preservation guarantee.
Adam moments follow ordinary summed-loss gradients; only parameter displacement
is corrected. Rejection restores both parameters and complete optimizer state.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'))
import t552_actor as A
P,G,torch,np=A.P,A.G,A.torch,A.np
read,save,sha,digest=A.read,A.save,A.sha,A.digest
Actor,frozen,make,state,load,fixed_hash,export,zero_export,feature_mean,location,cached_dataset=A.Actor,A.frozen,A.make,A.state,A.load,A.fixed_hash,A.export,A.zero_export,A.feature_mean,A.location,A.cached_dataset

def objectives(q,data,p):
    counts={k:sum(g['kind']==k for g in data) for k in ['restore','keep','neighbor']}
    losses={k:0. for k in counts};values={k:0. for k in counts}
    for g in data:
        loc=location(q,g)
        err=P.action_loss(loc,g['target']).mean(-1) if g['kind']=='restore' else (loc-g['target']).square().mean(-1)
        v=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/counts[g['kind']]
        losses[g['kind']]+=v;values[g['kind']]+=float(v.detach())
    return [losses['restore'],losses['keep']+losses['neighbor']],values

def directions(delta,h,zero_norm=1e-12):
    assert torch.isfinite(delta).all() and torch.isfinite(h).all()
    hn=float(h.norm());dn=float(delta.norm());dot=float(h@delta)
    active=hn>zero_norm and dot>0
    projected=delta-(dot/(hn*hn))*h if active else delta.clone()
    pn=float(projected.norm());scale=min(1.,pn/dn) if dn>0 else 1.
    scalar=delta*scale
    tol=1e-12*max(hn*dn,1e-30)
    if active:assert float(h@projected)<=tol
    assert pn<=dn+1e-13 and abs(float(scalar.norm())-pn)<=1e-12
    return projected,scalar,dict(active=active,retention_gradient_norm=hn,raw_delta_norm=dn,projected_delta_norm=pn,scalar_factor=scale,raw_retention_dot=dot,projected_retention_dot=float(h@projected),scalar_retention_dot=float(h@scalar),zero_gradient_skipped=hn<=zero_norm)

def step(obj,losses,p,guard,event,k,label):
    q,opt=obj;guard();params=list(q.extra.parameters())
    h=torch.cat([x.reshape(-1) for x in torch.autograd.grad(losses[1],params,retain_graph=True)]).detach()
    theta=torch.cat([x.detach().reshape(-1) for x in params]).clone()
    opt.zero_grad(set_to_none=True);sum(losses).backward()
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    event(label+'_BEFORE',k);opt.step()
    delta=torch.cat([x.detach().reshape(-1) for x in params])-theta
    projected,scalar,meta=directions(delta,h,p['retention_zero_norm'])
    chosen={'project':projected,'scalar':scalar}[p['arm']]
    with torch.no_grad():
        offset=0
        for x in params:x.copy_((theta[offset:offset+x.numel()]+chosen[offset:offset+x.numel()]).reshape_as(x));offset+=x.numel()
    actual=torch.cat([x.detach().reshape(-1) for x in params])-theta
    assert torch.allclose(actual,chosen,atol=1e-16,rtol=1e-10)
    event(label+'_AFTER',k)
    return dict(norm=norm,displacement=dict(**meta,applied_norm=float(actual.norm()),applied_retention_dot=float(h@actual)),adam_moments='ORDINARY_SUM_LOSS_RETAINED')

def bc(obj,data,pairs,p,guard,event,steps=None):
    fixed=fixed_hash(obj[0]);trace=[]
    for g in obj[1].param_groups:g['lr']=p['bc_lr']
    for k in range(p['bc_steps'] if steps is None else steps):
        losses,v=objectives(obj[0],data,p);loss_value=sum(float(x.detach()) for x in losses)
        meta=step(obj,losses,p,guard,event,k,'BC');trace.append(dict(step=k,loss=loss_value,**v,**meta))
    assert fixed_hash(obj[0])==fixed
    return trace

def rl_or_control(obj,arrays,data,pairs,p,guard,event):
    sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'))
    import t519_learning as Credit
    q,opt=obj;items,diag=Credit.peer_batch(obj,arrays,p);cached=[]
    for b in items:f,base=feature_mean(q,b['x']);cached.append(dict(b,f=f,base=base))
    fixed=fixed_hash(q);trace=[]
    for g in opt.param_groups:g['lr']=p['rl_lr']
    for k in range(p['rl_steps']):
        losses,v=objectives(q,data,p);policy=0.
        for b in cached:
            ratio=(P.L.density(location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            policy-=torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cached)
        total=[losses[0]+policy,losses[1]];pl=float(policy.detach());lv=sum(float(x.detach()) for x in total)
        meta=step(obj,total,p,guard,event,k,'RL')
        with torch.no_grad():
            z=[P.L.density(location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cached]
            kl=sum(float((x.exp()-1-x).mean()) for x in z)/len(z)
        trace.append(dict(step=k,total_loss=lv,policy_loss=pl,KL=kl,**v,**meta))
        if kl>p['target_kl']:break
    assert fixed_hash(q)==fixed
    return dict(trace=trace,peer_batch=diag,RL_Adam=len(trace),BC_control_Adam=0,new_Value=0)
