"""Three fixed actuator-credit ablations with native per-ID acceptance.

Joint keep-priority raw-gradient projection is PCGrad-inspired, not a theorem
for mature Adam or closed-loop safety. All data are supervised historical data.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t509_two_goal_joint'));import t509_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest
make,groups,measure=B.make,B.groups,B.measure

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;params=A.variables(q);fixed=A.fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());before=[int(ao.state[z]['step']) for z in params]
    mode=p['arm'];assert mode in ['drive_only','steer_only','joint_keep_priority']
    active=list(range(4)) if mode=='drive_only' else list(range(4,8)) if mode=='steer_only' else list(range(8))
    inactive={i:digest(dict(parameter=params[i],Adam=ao.state[params[i]])) for i in range(8) if i not in active}
    initial=measure(q,data,p);trace=[]
    def gradients(kind):
        ao.zero_grad(set_to_none=True);total=0.
        for g in data:
            if g['kind']!=kind:continue
            n=sum(h['kind']==kind and h['family']==g['family'] for h in data)
            loss=B.losses(q,g,p)/(2*n);loss.backward();total+=float(loss.detach())
        return [z.grad.detach().clone() if z.grad is not None else torch.zeros_like(z) for z in params],total
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();repair,rl=gradients('repair');keep,kl=gradients('keep');dot=sum(float((repair[i]*keep[i]).sum()) for i in active);keep_norm=sum(float(keep[i].square().sum()) for i in active)
        project=mode=='joint_keep_priority' and dot<0 and keep_norm>1e-24
        coef=dot/keep_norm if project else 0.
        ao.zero_grad(set_to_none=True)
        for i in active:params[i].grad=.5*(repair[i]-coef*keep[i])+.5*keep[i]
        norm=float(torch.nn.utils.clip_grad_norm_([params[i] for i in active],1.,error_if_nonfinite=True))
        old=[params[i].detach().clone() for i in active];event('BEFORE',k);ao.step();event('AFTER',k)
        displacement=[params[i].detach()-z for i,z in zip(active,old)]
        trace.append(dict(step=k,repair_loss=rl,keep_loss=kl,raw_gradient_dot=dot,keep_gradient_norm2=keep_norm,repair_projection_coefficient=coef,gradient_norm=norm,actual_Adam_displacement_dot_keep_gradient=sum(float((d*keep[i]).sum()) for i,d in zip(active,displacement))))
        assert A.fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        for i,h in inactive.items():assert digest(dict(parameter=params[i],Adam=ao.state[params[i]]))==h
    for i,z in enumerate(params):assert int(ao.state[z]['step'])==before[i]+(len(trace) if i in active else 0)
    return dict(arm=mode,Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=measure(q,data,p),trace=trace,active_parameter_indices=active,trainable_parameters=sum(params[i].numel() for i in active),frozen_inactive_parameters_and_mature_Adam_exact=True)
