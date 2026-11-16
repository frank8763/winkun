"""One change: add task-balanced fresh old-task PPO gradients to the same Actor."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t581_preserved_t550_rl'))
import t581_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
make,state,load,fixed,feature,loc=Prior.make,Prior.state,Prior.load,Prior.fixed,Prior.feature,Prior.loc
references,export,frozen=Prior.references,Prior.export,Prior.frozen


def fit(obj,arrays,data,p,arm,guard,event,steps=None):
    assert arm in p['arms'] and len(arrays)==24
    q,opt=obj;body=fixed(q);groups=[];diagnostics=[]
    # Four peers per identical task. No cross-task return baseline or RMS.
    for task in range(6):
        items,diag=Prior.Credit.peer_batch(obj,arrays[4*task:4*task+4],p)
        cache=[]
        for b in items:
            f,base=feature(q,b['x']);cache.append(dict(b,f=f,base=base))
        groups.append(cache);diagnostics.append(dict(task=p['training_jobs'][task]['id'],peers=diag))
    trace=[]
    for k in range(p['rl_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);new_pl=old_pl=keep=0.
        for task,cache in enumerate(groups):
            if arm=='NEW_ONLY' and task>0:continue
            # New-task coefficient is exactly one in both arms. The old family
            # has total coefficient one, shared equally by five tasks.
            weight=1. if task==0 else .2*p.get('old_family_weight',1.)
            for b in cache:
                z=loc(q,b['f'],b['base']);ratio=(P.L.density(z,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
                loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()*weight/4
                loss.backward()
                if task==0:new_pl+=float(loss.detach())
                else:old_pl+=float(loss.detach())
        for g in data:
            v=.5*((loc(q,g['f'],g['base'])-g['target'])/q.head.weight.new_tensor(p['raw_sigma'])).square().mean()/len(data)
            (p['keep_weight']*v).backward();keep+=float(v.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        with torch.no_grad():
            kls=[]
            for cache in groups:
                zz=[P.L.density(loc(q,b['f'],b['base']),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cache]
                kls.append(sum(float((v.exp()-1-v).mean()) for v in zz)/4)
        trace.append(dict(new_policy_loss=new_pl,old_family_policy_loss=old_pl,keep_loss=keep,gradient_norm=norm,task_KLs=kls))
        # Both arms use the same old+new density checks and stopping rule.
        if max(kls)>p['target_kl']:break
    assert fixed(q)==body
    return dict(arm=arm,Actor_Adam=len(trace),Critic_Adam=0,trace=trace,task_peer_batches=diagnostics,
        frozen_T550_body_hash=body,new_task_loss_coefficient=1.,old_family_loss_coefficient=0. if arm=='NEW_ONLY' else p.get('old_family_weight',1.),
        shared24_same_initial_policy_samples=True,not_CLEAR_replication=True)
