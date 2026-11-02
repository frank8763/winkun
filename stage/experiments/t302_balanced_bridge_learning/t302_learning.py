"""Own actual LEFT5.5m/RIGHT6m data; fixed side balance and native block gate."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t299_conditional_goal_likelihood'))
from t299_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t291_hard_guard_ablation'))
from t291_codec import unpack_full

def auxiliary(plan):
    items=[]
    for s in plan['auxiliary_sources']:
        assert C.sha(s['path'])==s['sha256']
        with np.load(s['path']) as z:a={k:z[k].copy() for k in z.files}
        if 'last_next_state' in a:a=unpack_full(a)
        assert a['states'].shape[1]==a['next_states'].shape[1]==120 and np.array_equal(a['end_flags'][-1],[0,1,0]) and not a['end_flags'][:-1].any()
        items.append((torch.tensor(a['states']),torch.tensor(a['actions'])))
    assert len(items)==15;return items

def task_weights():return [.5/5]*5+[.5/10]*10

def gradient_scale(actor,items):
    params=[q for q in actor.parameters() if q.requires_grad];report=[]
    for kind in ['clipped_mode_MSE','conditional_AR_NLL']:
        actor.zero_grad()
        for weight,item in zip(task_weights(),items):
            x,y=item
            for k in range(0,len(x),32):
                ix=np.arange(k,min(k+32,len(x)));loss=(actor.step(x[ix])[0]-y[ix]).square().sum()/2 if kind=='clipped_mode_MSE' else conditional_nll(actor,item,ix).sum()
                (loss*weight/len(x)).backward()
        g=torch.cat([torch.zeros_like(q).reshape(-1) if q.grad is None else q.grad.reshape(-1) for q in params]).detach().double();assert torch.isfinite(g).all();report.append(dict(objective=kind,norm=float(g.norm()),sha256=S.digest(g)))
    actor.zero_grad();factor=report[0]['norm']/report[1]['norm'];assert np.isfinite(factor) and factor>0
    return dict(factor=factor,objectives=report,rule='Parent full15trajectories sideequal .5LEFT/.5RIGHT, within side trajectoryequal, modeMSE / conditionalNLL gradient norm ratio fixed before any new native/training; not adaptive.',source_policy_hash=S.digest(actor.state_dict()),native_steps=0,optimizer_calls=0)

def supervised_metrics(actor,items,x,y,offset):
    errors=[];nll=[]
    with torch.no_grad():
        for item in items:
            a,b=item;errors.append(float((actor.step(a)[0]-b).square().mean()));nll.append(float(torch.cat([conditional_nll(actor,item,np.arange(k,min(k+32,len(a)))) for k in range(0,len(a),32)]).mean()))
        old,mx=anchor_measure(actor,x,y,offset)
    return dict(side_equal_MSE=float(np.dot(task_weights(),errors)),side_equal_conditional_NLL=float(np.dot(task_weights(),nll)),LEFT_MSE=float(np.mean(errors[:5])),RIGHT_MSE=float(np.mean(errors[5:])),per_trajectory_MSE=errors,per_trajectory_NLL=nll,old_balanced_MSE=float(old),old_max_error=float(mx))

def supervised_step(actor,opt,items,x,y,offset,plan,k):
    rng=np.random.default_rng(plan['supervised_seed']+k);ix=[rng.integers(0,len(a),4) for a,b in items];oi=np.concatenate([rng.integers(offset[t],offset[t+1],4) for t in range(8)])
    fit=sum(weight*conditional_nll(actor,item,i).mean() for weight,item,i in zip(task_weights(),items,ix));keep=plan['retention_weight']*(actor.step(x[oi])[0]-y[oi]).square().mean();loss=plan['auxiliary_NLL_scale']*fit+keep
    opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_([q for q in actor.parameters() if q.requires_grad],1.,error_if_nonfinite=True);opt.step()
    return dict(step=k,objective=float(loss.detach()),conditional_NLL=float(fit.detach()),old_weighted_MSE=float(keep.detach()),gradient_norm=float(gn))

def supervised_blocks(actor,shadow,opt,shadow_opt,items,x,y,offset,plan,guard,native_gate,record):
    initial=supervised_metrics(actor,items,x,y,offset);events=[];attempted=retained=0
    for block in range(plan['supervised_blocks']):
        before=copy.deepcopy(actor.state_dict());bo=copy.deepcopy(opt.state_dict());first=block*plan['supervised_block_steps'];last=None
        try:
            for k in range(first,first+plan['supervised_block_steps']):
                guard();a=supervised_step(actor,opt,items,x,y,offset,plan,k);b=supervised_step(shadow,shadow_opt,items,x,y,offset,plan,k);assert a==b;attempted+=1;last=a
            ph=S.digest(actor.state_dict());oh=S.digest(opt.state_dict());assert ph==S.digest(shadow.state_dict()) and oh==S.digest(shadow_opt.state_dict());native=native_gate(actor);accept=native['all_safe']
            if not accept:
                actor.load_state_dict(before);shadow.load_state_dict(before);opt.load_state_dict(bo);shadow_opt.load_state_dict(bo)
            else:retained+=plan['supervised_block_steps']
            event=dict(block=block,steps_attempted=plan['supervised_block_steps'],accepted=accept,proposal_policy_hash=ph,proposal_optimizer_hash=oh,retained_policy_hash=S.digest(actor.state_dict()),retained_optimizer_hash=S.digest(opt.state_dict()),native_old20=native,last_step=last);events.append(event);record(event)
            if not accept:break
        except BaseException:
            actor.load_state_dict(before);shadow.load_state_dict(before);opt.load_state_dict(bo);shadow_opt.load_state_dict(bo);record(dict(block=block,interrupted=True,retained_policy_hash=S.digest(actor.state_dict()),retained_optimizer_hash=S.digest(opt.state_dict()),attempted_steps=attempted));raise
    final=supervised_metrics(actor,items,x,y,offset)
    return dict(attempted_steps=attempted,retained_steps=retained,live_calls=attempted,shadow_calls=attempted,initial=initial,final=final,blocks=events,policy_hash=S.digest(actor.state_dict()),optimizer_hash=S.digest(opt.state_dict()),NLL_scale=plan['auxiliary_NLL_scale'],metrics_pass=final['side_equal_conditional_NLL']<=initial['side_equal_conditional_NLL'],scope='Sideequal ownLEFT5.5m/RIGHT6m conditionalNLL selfsupervision; nativeold20 accepts wholeblockelseActorAdamrollback and stop. No teacher/route/onpolicyclaim. Any following original6mRL uses freshAdam and newbehavior. Metrics diagnostic only; realold20 andLEFTbridgeautonomy gate followingRL.')
