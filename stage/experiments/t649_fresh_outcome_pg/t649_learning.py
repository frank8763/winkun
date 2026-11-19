"""Fresh original-start terminal-outcome clipped PG vs shared success BC.

Critic-free episodic score estimator, not full PPO/GAE or standard SIL.
No success filtering for PG; task LOO independent of each scored trajectory.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t646_offline_night'))
import t646_learning as B
L,P,E,torch,np=B.L,B.P,B.E,B.torch,B.np
state,load,export=B.state,B.load,B.export
read,save,sha,digest=B.read,B.save,B.sha,B.digest


def make(p,device='cuda'):return B.make(dict(p,arm='full_command'),device)


def outcome(row):
    return 1. if row['success'] and not row['collision'] and not row.get('field_boundary',False) else 0.


def build(q,groups,p):
    result=[];diag=[];device=next(q.parameters()).device
    for task,group in groups:
        assert len(group)==4;rewards=[outcome(row) for a,row in group];parts=[]
        for i,(a,row) in enumerate(group):
            assert a['stochastic'].all() and not a['ownership'].any()
            x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
            with torch.no_grad():
                loc=q.sequence(x);assert float((loc-torch.tensor(a['locations'],device=device)).abs().max())<1e-10
                assert float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())<1e-9
            # No normalization by own (action-dependent) episode length or
            # by sample standard deviation. Constant score_scale changes only
            # gradient scale. Task weight is exactly 1/number of groups.
            baseline=sum(rewards[j] for j in range(4) if j!=i)/3
            parts.append(dict(id=task,x=x,raw=raw,atom=atom,old=old,initial=loc.detach(),
                actions=torch.tensor(a['actions'],device=device).double(),adv=rewards[i]-baseline,success=bool(rewards[i]),source_sha256=row['sha256']))
        result.append(parts);diag.append(dict(id=task,outcomes=rewards,advantages=[v['adv'] for v in parts],independent_peer_baseline=True,informative=len(set(rewards))>1))
    return result,diag


def pg_objective(q,groups,p):
    loss=next(q.parameters()).new_zeros(())
    for group in groups:
        for b in group:
            loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            ratio_adv=ratio*b['adv'];clip_adv=ratio.clamp(1-p['ratio_clip'],1+p['ratio_clip'])*b['adv']
            loss-=torch.minimum(ratio_adv,clip_adv).sum()/(len(groups)*4*p['score_scale'])
    return loss


def bc_objective(q,groups,p):
    active=[g for g in groups if any(b['success'] for b in g)];loss=next(q.parameters()).new_zeros(())
    assert active
    for group in active:
        good=[b for b in group if b['success']]
        for b in good:
            loc=q.sequence(b['x']);err=P.action_loss(loc,b['actions']).mean(-1)/.2**2
            loss+=(.5*err.mean()+.5*err.topk(max(1,(len(err)+4)//5)).values.mean())/(len(active)*len(good))
    return loss


def keep_backward(q,old,p,backward=True):
    total=0.;device=next(q.parameters()).device
    # Full384 equal histories EVERY update; accumulation saves GPU memory.
    for off in range(0,len(old),p['old_batch']):
        batch=old[off:off+p['old_batch']];pred=B.evaluate_sequences(q,batch,device)
        loss=sum(P.action_loss(v,g['target'].to(device).clamp(-1,1)).mean() for v,g in zip(pred,batch))/(len(old)*.2**2)
        if backward:(p['keep_weight']*loss).backward()
        total+=float(loss.detach())
    return total


@torch.no_grad()
def trust(q,groups,p):
    return [sum(float((.5*((q.sequence(b['x'])-b['initial'])/.2).square().sum(-1)).mean()) for b in g)/4 for g in groups]


def fit(obj,old,groups,p,guard,event):
    q,opt=obj;items,diag=build(q,groups,p);trace=[]
    for k in range(p['fit_steps']):
        guard();before=state(obj);opt.zero_grad(set_to_none=True)
        # Accumulate task losses without keeping all76 recurrent graphs alive.
        main=0.
        if p['arm']=='pg':
            for group in items:
                loss=pg_objective(q,[group],p)/len(items);loss.backward();main+=float(loss.detach())
        else:
            active=[g for g in items if any(b['success'] for b in g)]
            for group in active:
                loss=bc_objective(q,[group],p)/len(active);loss.backward();main+=float(loss.detach())
        keep=keep_backward(q,old,p);variables=[v for v in q.parameters() if v.requires_grad]
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        kl=trust(q,items,p);okay=bool(np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL'])
        if not okay:load(obj,before)
        assert all(torch.isfinite(v).all() for v in variables)
        trace.append(dict(step=k,learning_loss=main,keep=keep,gradient_norm=norm,per_task_latent_KL=kl,accepted=okay))
        # Stop after a rejected proposal, no repeated identical proposal.
        if not okay:break
    return dict(Actor_Adam=len(trace),accepted=sum(v['accepted'] for v in trace),arm=p['arm'],trace=trace,
       diagnostics=diag,full384_each_update=True,full_causal_history=True,critic_updates=0,
       objective='mean task safe GOAL probability; divided by fixed score_scale' if p['arm']=='pg' else 'success-only command BC',
       not_standard_value_GAE_PPO=True)
