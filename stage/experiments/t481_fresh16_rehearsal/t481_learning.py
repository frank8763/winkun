"""Same old-task rehearsal pilot; 16 fresh episodes, equal episode weights.

No optimizer step at import. Retention/replay objectives are the exact frozen
T480 functions; only the fresh collection size and its estimator change.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t480_old_task_rehearsal'))
import t480_learning as B
P,E,M,R,torch,np=B.P,B.E,B.M,B.R,B.torch,B.np
read,save,sha,digest=B.read,B.save,B.sha,B.digest
make,references,memory,keep_loss,replay_losses=B.make,B.references,B.memory,B.keep_loss,B.replay_losses


def batch(obj,arrays,p):
    q,v,ao,vo=obj;device=next(q.parameters()).device
    assert len(arrays)==16 and p['fresh_groups']==[0]*8+[1]*8
    items=[]
    for i,a in enumerate(arrays):
        assert not a['ownership'].any() and a['stochastic'].all()
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device)
        atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
        with torch.no_grad():
            loc=q.sequence(x);err=float((loc-loc.new_tensor(a['locations'])).abs().max())
            lp=float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())
            assert err<1e-10 and lp<1e-9,(err,lp)
            vv=v(torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]),device=device)).cpu().numpy()
        mask=np.ones((len(x),1),bool);mask[-1]=False
        adv,target=P.L.physical_gae(a['rewards'][:,None],vv[:-1,None],vv[1:,None],
            a['native_counts'].astype(np.int64)[:,None],mask,mask,gamma_native=p['gamma_native'],lambda_native=1.)
        items.append(dict(x=x,raw=raw,atom=atom,old=old,adv=adv[:,0],target=old.new_tensor(target[:,0]),
                          group=p['fresh_groups'][i],raw_adv_mean=float(adv.mean()),density_error=[err,lp]))
    diagnostics=[]
    for group in [0,1]:
        selected=[b for b in items if b['group']==group]
        assert len(selected)==8
        mean=float(np.mean([b['adv'].mean() for b in selected]))
        second=float(np.mean([np.mean(b['adv']**2) for b in selected]))
        sd=math.sqrt(max(0.,second-mean*mean))+1e-8
        for b in selected:
            z=(b['adv']-mean)/sd
            diagnostics.append(dict(group=group,raw_mean=b['raw_adv_mean'],normalized_mean=float(z.mean()),shared_mean=mean,shared_sd=sd))
            b['adv']=b['old'].new_tensor(z)
    return items,diagnostics


def epoch(obj,items,refs,data,p,k,guard,event):
    guard();q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
    assert len(items)==16
    for b in items:
        ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(items)
        loss.backward();pg+=float(loss.detach())
        loss=.5*(v(b['x'])-b['target']).square().mean()/len(items)
        loss.backward();vl+=float(loss.detach())
    keep,old,actual,own=keep_loss(q,refs,k);(p['keep_weight']*keep).backward()
    al,cl,stats=replay_losses(q,v,data,p)
    (p['replay_actor_weight']*al).backward();(p['replay_value_weight']*cl).backward()
    an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
    vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
    event('B','BEFORE',k);ao.step();vo.step();event('B','AFTER',k)
    with torch.no_grad():
        kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/len(items)
    return dict(epoch=k,policy_loss=pg,value_loss=vl,keep=float(keep.detach()),
        old211_keep=float(old.detach()),actual_old8_keep=float(actual.detach()),own8_keep=float(own.detach()),
        replay_actor=float(al.detach()),replay_value=float(cl.detach()),replay_stats=stats,
        actor_norm=an,value_norm=vn,KL=kl,fresh_episodes=len(items))


def fit(obj,arrays,p,refs,data,guard,event,steps=None):
    assert digest(P.state(*obj))==p['initial']['joint_hash']
    items,diagnostics=batch(obj,arrays,p)
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        trace.append(epoch(obj,items,refs,data,p,k,guard,event))
        if trace[-1]['KL']>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diagnostics,
        old_replay_NOT_fresh_PPO=True,all_old_outcomes_included=True,
        single_pilot_NOT_matched_ablation=True,fresh_episode_weight=1/16)
