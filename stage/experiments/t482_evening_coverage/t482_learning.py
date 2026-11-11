"""Fixed targeted replay; conditional mean-versus-worst-history pilot.

No runtime teacher, no optimizer steps at import. Uses exactly the T481
16-episode fresh estimator with two explicitly registered finite anchors.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t481_fresh16_rehearsal'))
import t481_learning as F
B,P,E,M,R,torch,np=F.B,F.P,F.E,F.M,F.R,F.torch,F.np
read,save,sha,digest=F.read,F.save,F.sha,F.digest
make,references,memory,replay_losses=F.make,F.references,F.memory,F.replay_losses
batch=F.batch


def aggregate(values,mode):
    assert values.ndim==1 and len(values)==8
    if mode=='targeted_mean':return values.mean()
    assert mode=='targeted_max'
    # First-index argmax has a fixed tie rule. At exact zero all anchor
    # gradients are also zero. No softmax/temperature or fitted weight.
    return values[torch.argmax(values.detach())]


def keep_loss(q,refs,k):
    _,old,own=B.H.keep_loss(q,refs,k)
    sd=next(q.parameters()).new_tensor(q.p['raw_sigma'])
    losses=torch.stack([.5*((q.sequence(g['x'])-g['target'])/sd).square().mean()
                        for g in refs['actual_old']])
    current=aggregate(losses,q.p['anchor_mode'])
    return .5*(.5*(old+current)+own),old,current,own


def epoch(obj,items,refs,data,p,k,guard,event):
    guard();q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
    assert len(items)==16
    for b in items:
        ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/16
        loss.backward();pg+=float(loss.detach())
        loss=.5*(v(b['x'])-b['target']).square().mean()/16
        loss.backward();vl+=float(loss.detach())
    keep,old,actual,own=keep_loss(q,refs,k);(p['keep_weight']*keep).backward()
    al,cl,stats=replay_losses(q,v,data,p)
    (p['replay_actor_weight']*al).backward();(p['replay_value_weight']*cl).backward()
    an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
    vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
    event('B','BEFORE',k);ao.step();vo.step();event('B','AFTER',k)
    with torch.no_grad():
        kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/16
    return dict(epoch=k,policy_loss=pg,value_loss=vl,keep=float(keep.detach()),old211_keep=float(old.detach()),
        actual_targeted8_keep=float(actual.detach()),own8_keep=float(own.detach()),replay_actor=float(al.detach()),
        replay_value=float(cl.detach()),replay_stats=stats,actor_norm=an,value_norm=vn,KL=kl,
        fresh_episodes=16,anchor_mode=p['anchor_mode'])


def fit(obj,arrays,p,refs,data,guard,event,steps=None):
    assert digest(P.state(*obj))==p['initial']['joint_hash']
    items,diagnostics=batch(obj,arrays,p)
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        trace.append(epoch(obj,items,refs,data,p,k,guard,event))
        if trace[-1]['KL']>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diagnostics,
        old_replay_NOT_fresh_PPO=True,all_old_outcomes_included=True,single_pilot_NOT_matched_ablation=True,
        fresh_episode_weight=1/16,anchor_mode=p['anchor_mode'])
