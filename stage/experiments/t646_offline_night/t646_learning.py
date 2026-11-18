"""Matched full-history BC contrasts: heads/entire actor and command/raw keep."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import t645_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
state,load,export=Base.state,Base.load,Base.export


def make(p,device='cuda'):
    q=L.frozen(p,p['initial']['path']).to(device).train()
    if p['arm']=='heads_command':q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    else:q.requires_grad_(True)
    opt=torch.optim.Adam([v for v in q.parameters() if v.requires_grad],lr=p['fit_lr'],foreach=False,fused=False)
    assert not opt.state
    return q,opt


def pack(data,device):return data


def correction(p,a,k):
    assert 0<k<len(a['states'])
    return dict(x=torch.tensor(a['states']),target=torch.tensor(a['actions']).double(),first=k)


def evaluate_sequences(q,groups,device):
    lengths=[len(g['x']) for g in groups]
    x=torch.nn.utils.rnn.pad_sequence([g['x'].to(device) for g in groups])
    loc=q.sequence(x)
    return [loc[:n,i] for i,n in enumerate(lengths)]


def objective(q,old,new,p):
    device=next(q.parameters()).device
    combined=old+new;pred=evaluate_sequences(q,combined,device)
    keep=pred[0].new_zeros(())
    for loc,g in zip(pred[:len(old)],old):
        target=g['target'].to(device)
        err=(loc-target).square() if p['arm']=='full_raw' else P.action_loss(loc,target.clamp(-1,1))
        keep+=err.mean()/len(old)/.2**2
    counts={g['id']:sum(v['id']==g['id'] for v in new) for g in new}
    repair=keep.new_zeros(())
    for loc,g in zip(pred[len(old):],new):
        k=g['first'];err=P.action_loss(loc[k:],g['target'][k:].to(device)).mean(-1)/.2**2
        v=.5*err.mean()+.5*err.topk(max(1,(len(err)+4)//5)).values.mean()
        repair+=v/(len(counts)*counts[g['id']])
    return repair+p['keep_weight']*keep,repair,keep


def fit(obj,old,new,p,guard,event):
    q,opt=obj;trace=[];n=len(old);assert n==384
    frozen={k:copy.deepcopy(v) for k,v in q.state_dict().items() if not k.startswith(('drive_adapter.','steer_adapter.'))}
    for k in range(p['fit_steps']):
        guard();indices=[(k*p['old_batch']+i)%n for i in range(p['old_batch'])];batch=[old[i] for i in indices]
        opt.zero_grad(set_to_none=True);loss,r,s=objective(q,batch,new,p);loss.backward()
        variables=[v for v in q.parameters() if v.requires_grad]
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert all(torch.isfinite(v).all() for v in variables)
        trace.append(dict(step=k,old_indices=indices,loss=float(loss.detach()),repair=float(r.detach()),keep=float(s.detach()),norm=norm))
    if p['arm']=='heads_command':assert all(torch.equal(q.state_dict()[k],v) for k,v in frozen.items())
    assert p['fit_steps']*p['old_batch']%384==0
    with torch.no_grad():
        reports=[]
        for g in new:
            loc=q.sequence(g['x'].to(next(q.parameters()).device));k=g['first']
            reports.append(dict(id=g['id'],source_sha256=g['source_sha256'],RMSE=float((loc[k:].clamp(-1,1)-g['target'][k:].to(loc.device)).square().mean().sqrt())))
    return dict(Actor_Adam=len(trace),Value_Adam=0,arm=p['arm'],full_current_history_BPTT=True,
       full384_complete_passes=p['fit_steps']*p['old_batch']//384,
       trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad),trace=trace,metrics=dict(corrections=reports))
