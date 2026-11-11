"""Wide8-only recurrent PPO: no old-task replay or auxiliary gradients."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t478_wide8_pooled_credit'))
import t478_learning as C
P,E,M,R,torch,np=C.P,C.E,C.M,C.R,C.torch,C.np
read,save,sha,digest=C.read,C.save,C.sha,C.digest
make=C.make

def fit(obj,arrays,p,guard,event,steps=None):
    assert digest(P.state(*obj))==p['initial']['joint_hash']
    q,v,ao,vo=obj;items,diag=uncentered_batch(obj,arrays,p)
    with torch.no_grad():targets=[q.sequence(b['x']).detach().clone() for b in items]
    anchors=[]
    cap=read(p['capture_summary_path']) if Path(p['capture_summary_path']).exists() else None
    if cap is not None or p.get('fixture_anchor_rows'):
        for row in (cap['result']['diagnostics'] if cap is not None else p['fixture_anchor_rows']):
            if row['job_id'] in p['anchor_ids'] or cap is None:
                assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
                a=E.N.S.arrays(row['path']);x=torch.tensor(a['states'],device=next(q.parameters()).device)
                with torch.no_grad():target=q.sequence(x).detach().clone()
                anchors.append((x,target))
        assert len(anchors)==2
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=keep=0.
        for b,target in zip(items,targets):
            loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4
            anchor=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/4
            (loss+p['keep_weight']*anchor).backward();pg+=float(loss.detach());keep+=float(anchor.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/4;loss.backward();vl+=float(loss.detach())
        for x,target in anchors:
            loss=.5*((q.sequence(x)-target)/x.new_tensor(p['raw_sigma'])).square().mean()/len(anchors)
            loss.backward();keep+=float(loss.detach())
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
        vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
        event('B','BEFORE',k);ao.step();vo.step();event('B','AFTER',k)
        with torch.no_grad():
            kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/4
        trace.append(dict(epoch=k,policy_loss=pg,value_loss=vl,eight_current_history_keep=keep,actor_norm=an,value_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diag,eight_only=True,oldtask_gradient_or_reference=False,balanced_two_left_two_right_reused=True,actual_success_history_anchors=len(anchors),uncentered_advantage_RMS=True)


def uncentered_batch(obj,arrays,p):
    items,original=C.batch(obj,arrays,p)
    with torch.no_grad():
        for b in items:b['adv']=(b['target']-obj[1](b['x'])).detach()
    diagnostics=[]
    for group in [0,1]:
        selected=[b for b in items if b['group']==group]
        rms=(sum(float(b['adv'].square().mean()) for b in selected)/len(selected))**.5+1e-8
        for b in selected:
            mean=float(b['adv'].mean());positive=float((b['adv']>0).double().mean())
            b['adv']=b['adv']/rms
            diagnostics.append(dict(group=group,raw_mean=mean,RMS=rms,normalized_mean=float(b['adv'].mean()),positive_fraction=positive,mean_subtraction=False))
    return items,diagnostics
