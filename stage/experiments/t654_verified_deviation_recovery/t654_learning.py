"""Matched nominal versus verified actual-departure recovery supervision.

Frozen backbone permits exact cached full-carry features; this is BC, not RL.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t652_parameter_scope_diagnosis'))
import t652_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
make,state,load,export=Base.make,Base.state,Base.load,Base.export
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest


def features(q,a,row,first=0,target=None):
    x=torch.tensor(a['states'])
    with torch.no_grad():
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];f=torch.cat([h,n],-1);logits=q.head(h)
        own=q.sequence(x);loc=location(q,dict(f=f,logits=logits))
        assert float((loc-own).abs().max())<1e-10
    t=torch.tensor(a['actions'],dtype=torch.float64) if target is None else target.detach().double()
    assert 0<=first<len(x)
    return dict(id=row['job_id'],f=f,logits=logits,target=t,initial=own.detach(),first=first,source_sha256=row['sha256'])


def location(q,g):
    f=g['f'];res=torch.cat([q.drive_adapter(f),q.steer_adapter(f)],-1)
    return 1.25*torch.tanh(g['logits']+.3*torch.tanh(res))


def pack(data,device):
    return dict(f=torch.cat([g['f'] for g in data]).to(device),logits=torch.cat([g['logits'] for g in data]).to(device),target=torch.cat([g['target'] for g in data]).to(device),
        weights=torch.cat([torch.full((len(g['f']),),1/(len(data)*len(g['f'])),dtype=torch.float64) for g in data]).to(device))


def keep_loss(q,g):return (P.action_loss(location(q,g),g['target'].clamp(-1,1)).mean(-1)*g['weights']).sum()/.2**2


def repair_loss(q,groups):
    loss=next(q.parameters()).new_zeros(())
    for g in groups:
        err=P.action_loss(location(q,g)[g['first']:],g['target'][g['first']:].clamp(-1,1)).mean(-1)/.2**2
        loss+=(.5*err.mean()+.5*err.topk(max(1,(len(err)+4)//5)).values.mean())/len(groups)
    return loss


def fit(obj,old,new,p,guard,event):
    q,opt=obj;body=L.fixed_hash(q);device=next(q.parameters()).device
    keep=pack(old,device);fields=pack(new['fields'],device)
    repairs=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in g.items()} for g in new[p['arm']]]
    assert len(old)==384 and len(new['fields'])==8 and len(repairs)==3
    trace=[]
    for k in range(p['fit_steps']):
        guard();before=state(obj);opt.zero_grad(set_to_none=True)
        a=keep_loss(q,keep);b=keep_loss(q,fields);c=repair_loss(q,repairs)
        loss=p['keep_weight']*a+p['new_keep_weight']*b+c;loss.backward()
        variables=[v for v in q.parameters() if v.requires_grad];norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        okay=all(bool(torch.isfinite(v).all()) for v in variables)
        if not okay:load(obj,before)
        trace.append(dict(step=k,loss=float(loss.detach()),old_keep=float(a.detach()),field_keep=float(b.detach()),repair=float(c.detach()),norm=norm,accepted=okay))
        assert L.fixed_hash(q)==body
        if not okay:assert digest(state(obj))==digest(before);break
    with torch.no_grad():metrics=dict(old_command_RMSE=float(keep_loss(q,keep).sqrt()*.2),field_command_RMSE=float(keep_loss(q,fields).sqrt()*.2),repair_loss=float(repair_loss(q,repairs)))
    return dict(Actor_Adam=len(trace),accepted=sum(v['accepted'] for v in trace),arm=p['arm'],trace=trace,metrics=metrics,
        trainable_parameters=11906,encoder_GRU_head_frozen_exact=True,full384_each_update=True,all8_field_histories_each_update=True,
        exact_full_carry_cached_features=True,new_RL=0,new_Critic=0,supervised_not_TaSIL_or_guarantee=True)
