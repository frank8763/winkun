"""Capped per-task weighting, fixed during a round, exact Actor/Adam rollback.

This is an engineering candidate, not GradNorm/CAGrad or a novel method claim.
"""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t280_core import *


def norm(g):return float(torch.linalg.vector_norm(g.double()))


def fixed_balance(gradients,seed,cap=32.,relative_noise_floor=1e-8):
    assert cap>=1 and 0<relative_noise_floor<1
    sizes=[norm(g) for g in gradients];reference=max(sizes)
    floor=max(reference*relative_noise_floor,1e-12)
    factors=[min(cap,reference/n) if n>floor else 1. for n in sizes]
    original,_=L.project(gradients,seed);weighted=[g*f for g,f in zip(gradients,factors)];new,_=L.project(weighted,seed)
    old_norm,new_norm=norm(original),norm(new)
    # Do not manufacture a direction when the original estimator is zero.
    common_scale=old_norm/new_norm if old_norm>1e-12 and new_norm>1e-12 else 1.
    assert all(1<=f<=cap for f in factors) and math.isfinite(common_scale)
    return dict(task_gradient_norms=sizes,noise_floor=floor,factors=factors,common_scale=common_scale,
        original_PCGrad_norm=old_norm,unscaled_weighted_PCGrad_norm=new_norm,
        rescaled_weighted_PCGrad_norm=norm(new*common_scale),weights_frozen_within_round=True,
        missing_or_roundoff_signal_not_amplified=True)


def balanced_closed_loop_update(p,opt,data,x,y,offset,plan,rnd,guard,native_gate,on_step,balance=None):
    rng=np.random.default_rng(plan['seed']+rnd);indices=np.concatenate([rng.integers(offset[t],offset[t+1],32) for t in range(8)]);sx=x[indices];sy=y[indices]
    params=[q for q in p.parameters() if q.requires_grad];reports=[];flat=lambda gs:torch.cat([g.reshape(-1) for g in gs])
    def raw_measure():
        lp,_=L.likelihood(p,data);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage']
        term=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights']
        ss=[term[data['task']==t].sum()/4/plan['surrogate_horizon_decisions'] for t in range(4)]
        ret=plan['retention_weight']*(p.step(sx)[0]-sy).square().mean();return ss,ret,delta
    if balance is None:
        guard();ss,_,_=raw_measure();initial=[flat(torch.autograd.grad(-s,params,retain_graph=True)).detach() for s in ss]
        balance=fixed_balance(initial,plan['seed']+100*rnd,plan['balance_cap'],plan['balance_relative_noise_floor'])
        del ss,initial
    def measure():
        ss,ret,delta=raw_measure();weighted=[s*f*balance['common_scale'] for s,f in zip(ss,balance['factors'])]
        surrogate=torch.stack(weighted).mean();return surrogate-ret,surrogate,delta,weighted,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();obj,_,_,ss,ret=measure();before=float(obj.detach())
        gs=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in ss];rg=flat(torch.autograd.grad(ret,params))
        g,conflicts=L.project(gs,plan['seed']+100*rnd+k);g+=rg;opt.zero_grad();pos=0
        for q in params:q.grad=g[pos:pos+q.numel()].reshape_as(q).clone();pos+=q.numel()
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as error:return dict(accept=False,numeric_domain_rejection=str(error))
            mse,mx=anchor_measure(q,x,y,offset);k3=(d.exp()-1-d).mean();dl=d.abs().max()
            numeric=bool(torch.isfinite(o)) and bool(torch.isfinite(mse)) and bool(torch.isfinite(mx)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio']
            native=native_gate(q) if numeric else None;accept=numeric and bool(native['all_safe'])
            return dict(accept=accept,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_k3=float(k3),max_abs_log_ratio=float(dl),
                balanced_anchor_MSE=float(mse),anchor_max=float(mx),would_pass_T257_function_bounds=float(mse)<=plan['max_anchor_MSE'] and float(mx)<=plan['max_anchor_error'],native_old_task_guard=native)
        r=L.guarded_adam_step(p,opt,evaluate,plan['step_factors']);r.update(step=k,gradient_norm=gn,pcgrad_conflicts=conflicts,
            policy_hash=S.digest(p.state_dict()),optimizer_hash=S.digest(opt.state_dict()),fixed_round_balance=balance);reports.append(r);on_step(r)
        if not r['accepted']:break
    return reports,balance


def paired_balanced_update(p,opt,shadow,shadow_opt,data,x,y,offset,plan,rnd,guard,native_gate,record):
    before=copy.deepcopy(p.state_dict());before_opt=copy.deepcopy(opt.state_dict())
    try:
        live,balance=balanced_closed_loop_update(p,opt,data,x,y,offset,plan,rnd,guard,native_gate,lambda r:record('live',r))
        audit,again=balanced_closed_loop_update(shadow,shadow_opt,data,x,y,offset,plan,rnd,guard,native_gate,lambda r:record('shadow',r))
        assert live==audit and balance==again and S.digest(p.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
        return live,audit
    except BaseException as error:
        p.load_state_dict(before);opt.load_state_dict(before_opt);record('rollback',dict(reason=type(error).__name__,restored_policy_hash=S.digest(p.state_dict()),restored_optimizer_hash=S.digest(opt.state_dict()),partial_round_not_retained=True));raise
