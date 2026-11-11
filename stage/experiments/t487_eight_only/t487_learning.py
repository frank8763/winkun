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
    q,v,ao,vo=obj;items,diag=C.batch(obj,arrays,p)
    with torch.no_grad():targets=[q.sequence(b['x']).detach().clone() for b in items]
    trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=keep=0.
        for b,target in zip(items,targets):
            loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4
            anchor=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/4
            (loss+p['keep_weight']*anchor).backward();pg+=float(loss.detach());keep+=float(anchor.detach())
            loss=.5*(v(b['x'])-b['target']).square().mean()/4;loss.backward();vl+=float(loss.detach())
        an=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True))
        vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
        event('B','BEFORE',k);ao.step();vo.step();event('B','AFTER',k)
        with torch.no_grad():
            kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/4
        trace.append(dict(epoch=k,policy_loss=pg,value_loss=vl,eight_current_history_keep=keep,actor_norm=an,value_norm=vn,KL=kl))
        if kl>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diag,eight_only=True,oldtask_gradient_or_reference=False)
