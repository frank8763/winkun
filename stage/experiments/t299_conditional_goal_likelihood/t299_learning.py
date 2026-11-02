"""Same fixed auxiliary data, corrected conditional likelihood with fixed scale.

Goal-conditioned self-supervision, not an IS estimator or on-policy RL. Native
policy/sampler unchanged; previous conditional mean remains differentiable.
"""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t298_self_goal_transfer'))
from t298_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t296_log_tail_support'))
import stable_density as D

def exact_mean(actor,x):
    return torch.cat([actor.step(row[None])[1] for row in x])

def conditional_nll(actor,item,index):
    x,y=item;index=np.asarray(index);previous=np.maximum(index-1,0)
    # Canonical batch1 used on both current/previous states, no shifted means,
    # detached previous-mean gradient or old-logp replacement.
    current=exact_mean(actor,x[index]);pm=exact_mean(actor,x[previous]);a=y[index];pa=y[previous]
    reset=torch.tensor(index==0,dtype=torch.bool)
    return -D.logp(current,a,pm,pa,reset)

def gradient_scale(actor,items):
    params=[q for q in actor.parameters() if q.requires_grad];results=[]
    for kind in ['clipped_mode_MSE','conditional_AR_NLL']:
        actor.zero_grad()
        for item in items:
            x,y=item
            for begin in range(0,len(x),32):
                ix=np.arange(begin,min(begin+32,len(x)))
                if kind=='clipped_mode_MSE':loss=(actor.step(x[ix])[0]-y[ix]).square().sum()/2
                else:loss=conditional_nll(actor,item,ix).sum()
                (loss/len(items)/len(x)).backward()
        values=[torch.zeros_like(q).reshape(-1) if q.grad is None else q.grad.reshape(-1) for q in params];g=torch.cat(values).detach().double();assert torch.isfinite(g).all();results.append(dict(objective=kind,gradient_norm=float(g.norm()),gradient_sha256=S.digest(g)))
    actor.zero_grad();factor=results[0]['gradient_norm']/results[1]['gradient_norm'];assert 0<factor<1 and np.isfinite(factor)
    return dict(rule='Frozen safeT255 full8trajectory-equal mean-gradient norm ratio: norm(clipped_mode_MSE)/norm(conditional_AR_NLL). Fixed once, not fit to native outcome; no clipping or adaptive rescale.',factor=factor,objectives=results,native_steps=0,optimizer_calls=0,source_policy_hash=S.digest(actor.state_dict()))

def supervised_metrics(actor,items,x,y,offset):
    with torch.no_grad():
        errors=[(actor.step(a)[0]-b).square().mean() for a,b in items]
        nll=[]
        for item in items:
            pieces=[conditional_nll(actor,item,np.arange(k,min(k+32,len(item[0])))) for k in range(0,len(item[0]),32)]
            nll.append(float(torch.cat(pieces).mean()))
        old,maxerr=anchor_measure(actor,x,y,offset)
    return dict(auxiliary_trajectory_equal_MSE=float(torch.stack(errors).mean()),per_trajectory_MSE=[float(e) for e in errors],auxiliary_equal_conditional_NLL=float(np.mean(nll)),per_trajectory_conditional_NLL=nll,old_balanced_MSE=float(old),old_max_error=float(maxerr))

def supervised_step(actor,opt,items,x,y,offset,plan,k):
    rng=np.random.default_rng(plan['supervised_seed']+k);ix=[rng.integers(0,len(a),8) for a,b in items]
    oi=np.concatenate([rng.integers(offset[t],offset[t+1],4) for t in range(8)]);oldx=x[oi];oldy=y[oi]
    # Equal trajectory and decision weighting exactly matches the previous MSE
    # sampling. Only add each sampled decision's actual previous context.
    fit=torch.stack([conditional_nll(actor,item,i).mean() for item,i in zip(items,ix)]).mean()
    scaled=fit*plan['auxiliary_NLL_scale'];keep=plan['retention_weight']*(actor.step(oldx)[0]-oldy).square().mean();loss=scaled+keep
    opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_([v for v in actor.parameters() if v.requires_grad],1.,error_if_nonfinite=True);opt.step()
    return dict(step=k,objective=float(loss.detach()),auxiliary_conditional_NLL=float(fit.detach()),scaled_auxiliary_NLL=float(scaled.detach()),old_weighted_MSE=float(keep.detach()),gradient_norm=float(gn))

def supervised_pair(actor,shadow,opt,shadow_opt,items,x,y,offset,plan,guard,record):
    initial=supervised_metrics(actor,items,x,y,offset)
    for k in range(plan['supervised_steps']):
        guard();live=supervised_step(actor,opt,items,x,y,offset,plan,k);audit=supervised_step(shadow,shadow_opt,items,x,y,offset,plan,k);assert live==audit
        if (k+1)%100==0 or k+1==plan['supervised_steps']:
            assert S.digest(actor.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
            record(dict(**live,policy_hash=S.digest(actor.state_dict()),supervised_optimizer_hash=S.digest(opt.state_dict())))
    final=supervised_metrics(actor,items,x,y,offset)
    return dict(steps=plan['supervised_steps'],live_calls=plan['supervised_steps'],shadow_calls=plan['supervised_steps'],initial=initial,final=final,policy_hash=S.digest(actor.state_dict()),optimizer_hash=S.digest(opt.state_dict()),NLL_scale=plan['auxiliary_NLL_scale'],metrics_pass=final['auxiliary_trajectory_equal_MSE']<=.5*initial['auxiliary_trajectory_equal_MSE'],scope='Fixed500 own-action conditional-AR log-likelihood self-supervision, NOT policy gradient/HPG/on-policy RL. Frozen full-data gradient-norm scale and oldmodeMSE100; canonical current/previousbatch1 paths. No new RL in this protocol, no changed native sampler/Actor/auxgoals. Supervised Adam reconstructed independently, fresh RL Adam saved only as empty reference, no continuation authorized by this directory.')
