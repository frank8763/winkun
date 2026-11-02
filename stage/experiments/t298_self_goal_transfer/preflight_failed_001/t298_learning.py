"""Explicit own-trajectory supervised auxiliary phase; subsequent RL separate."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t282_capped_balance'))
from balanced_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import *
sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'))
from store import resolve

def auxiliary(plan):
    items=[]
    for source in plan['auxiliary_sources']:
        with np.load(source['path']) as z:
            x=z['alternate_states'].copy();y=z['alternate_actions'].copy();flags=z['alternate_end_flags'].copy();counts=z['alternate_native_counts'].copy()
        assert x.shape[1]==120 and x.shape[0]==len(y) and np.array_equal(flags[-1],[0,1,0]) and not flags[:-1].any()
        assert len(x)>=2 and int(counts[-1])<100
        items.append((torch.tensor(x),torch.tensor(y)))
    assert len(items)==8;return items

def supervised_metrics(actor,items,x,y,offset):
    with torch.no_grad():
        errors=[(actor.step(a)[0]-b).square().mean() for a,b in items]
        old,maxerr=anchor_measure(actor,x,y,offset)
    return dict(auxiliary_trajectory_equal_MSE=float(torch.stack(errors).mean()),per_trajectory_MSE=[float(e) for e in errors],old_balanced_MSE=float(old),old_max_error=float(maxerr))

def supervised_step(actor,opt,items,x,y,offset,plan,k):
    rng=np.random.default_rng(plan['supervised_seed']+k)
    # Each whole trajectory contributes equally; all eight are consumed developer
    # data, never heldout evidence. Matched shadow uses exact same indices.
    ix=[rng.integers(0,len(a),8) for a,b in items]
    xx=torch.cat([a[i] for (a,b),i in zip(items,ix)]);yy=torch.cat([b[i] for (a,b),i in zip(items,ix)])
    oi=np.concatenate([rng.integers(offset[t],offset[t+1],4) for t in range(8)]);oldx=x[oi];oldy=y[oi]
    fit=(actor.step(xx)[0]-yy).square().mean();keep=plan['retention_weight']*(actor.step(oldx)[0]-oldy).square().mean();loss=fit+keep
    opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_([v for v in actor.parameters() if v.requires_grad],1.,error_if_nonfinite=True);opt.step()
    return dict(step=k,objective=float(loss.detach()),auxiliary_MSE=float(fit.detach()),old_weighted_MSE=float(keep.detach()),gradient_norm=float(gn))

def supervised_pair(actor,shadow,opt,shadow_opt,items,x,y,offset,plan,guard,record):
    initial=supervised_metrics(actor,items,x,y,offset);reports=[]
    for k in range(plan['supervised_steps']):
        guard();live=supervised_step(actor,opt,items,x,y,offset,plan,k);audit=supervised_step(shadow,shadow_opt,items,x,y,offset,plan,k)
        assert live==audit
        if (k+1)%100==0 or k+1==plan['supervised_steps']:
            assert S.digest(actor.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(shadow_opt.state_dict())
            row=dict(**live,policy_hash=S.digest(actor.state_dict()),supervised_optimizer_hash=S.digest(opt.state_dict()));reports.append(row);record(row)
    final=supervised_metrics(actor,items,x,y,offset)
    return dict(steps=plan['supervised_steps'],live_calls=plan['supervised_steps'],shadow_calls=plan['supervised_steps'],initial=initial,final=final,policy_hash=S.digest(actor.state_dict()),optimizer_hash=S.digest(opt.state_dict()),metrics_pass=final['auxiliary_trajectory_equal_MSE']<=.5*initial['auxiliary_trajectory_equal_MSE'],scope='Own-action goal-conditioned supervised regression, NOT policy-gradient/HPG/on-policy RL. Source trajectories already consumed, no independent test. Supervised Adam discarded before fresh RL Adam.')

def compact(a):
    # Remove only a deterministic duplicate: rewards = env_rewards - eta*cost.
    out={k:a[k] for k in ['actions','native_counts','end_flags','environment_rewards','aux_costs']}
    assert sum(v.nbytes for v in out.values())<=7840
    return out

def unpack_compact(a,eta):
    assert set(a)=={'actions','native_counts','end_flags','environment_rewards','aux_costs'}
    return dict(a,rewards=a['environment_rewards']-eta*a['aux_costs'])
