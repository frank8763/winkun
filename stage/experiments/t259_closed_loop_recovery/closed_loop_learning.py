"""Actual old-task rollout acceptance replaces hard old-action imitation bounds.

This is a finite-case engineering constraint, not a CPO implementation or proof.
The deployment graph and all learner settings remain the accepted120D Actor.
"""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t257_distance_policy_gradient'))
from distance_learning import *

def closed_loop_update(p,opt,data,x,y,offset,plan,rnd,guard,native_gate,on_step):
    rng=np.random.default_rng(plan['seed']+rnd);indices=np.concatenate([rng.integers(offset[t],offset[t+1],32) for t in range(8)]);sx=x[indices];sy=y[indices];params=[q for q in p.parameters() if q.requires_grad];reports=[];flat=lambda gs:torch.cat([g.reshape(-1) for g in gs])
    def measure():
        lp,_=L.likelihood(p,data);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];term=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights'];ss=[term[data['task']==t].sum()/4/plan['surrogate_horizon_decisions'] for t in range(4)];s=torch.stack(ss).mean();ret=plan['retention_weight']*(p.step(sx)[0]-sy).square().mean();return s-ret,s,delta,ss,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();obj,_,_,ss,ret=measure();before=float(obj.detach());gs=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in ss];rg=flat(torch.autograd.grad(ret,params));g,conflicts=L.project(gs,plan['seed']+100*rnd+k);g+=rg;opt.zero_grad();pos=0
        for q in params:q.grad=g[pos:pos+q.numel()].reshape_as(q).clone();pos+=q.numel()
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            mse,mx=anchor_measure(q,x,y,offset);k3=(d.exp()-1-d).mean();dl=d.abs().max();numeric=bool(torch.isfinite(o)) and bool(torch.isfinite(mse)) and bool(torch.isfinite(mx)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio']
            # Full old-function errors are reported, but not task-retention criteria.
            native=native_gate(q) if numeric else None;accept=numeric and bool(native['all_safe'])
            return dict(accept=accept,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_k3=float(k3),max_abs_log_ratio=float(dl),balanced_anchor_MSE=float(mse),anchor_max=float(mx),would_pass_T257_function_bounds=float(mse)<=plan['max_anchor_MSE'] and float(mx)<=plan['max_anchor_error'],native_old_task_guard=native)
        r=L.guarded_adam_step(p,opt,evaluate,plan['step_factors']);r.update(step=k,gradient_norm=gn,pcgrad_conflicts=conflicts,policy_hash=S.digest(p.state_dict()),optimizer_hash=S.digest(opt.state_dict()));reports.append(r);on_step(r)
        if not r['accepted']:break
    return reports

def paired_update(p,opt,shadow,shadow_opt,data,x,y,offset,plan,rnd,guard,native_gate,record):
    before=copy.deepcopy(p.state_dict());before_opt=copy.deepcopy(opt.state_dict())
    try:
        live=closed_loop_update(p,opt,data,x,y,offset,plan,rnd,guard,native_gate,lambda r:record('live',r))
        audit=closed_loop_update(shadow,shadow_opt,data,x,y,offset,plan,rnd,guard,native_gate,lambda r:record('shadow',r))
        assert live==audit and S.digest(p.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
        return live,audit
    except BaseException as error:
        p.load_state_dict(before);opt.load_state_dict(before_opt);record('rollback',dict(reason=type(error).__name__,restored_policy_hash=S.digest(p.state_dict()),restored_optimizer_hash=S.digest(opt.state_dict()),partial_round_not_retained=True));raise
