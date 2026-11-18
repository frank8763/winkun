"""Success-conditioned sampled-action supervision; no Critic or mixed PG."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t629_offline_night_relay'))
import t629_learning as M
torch,np=M.torch,M.np


def dataset(p,corrections,anchors,device='cuda'):
    values=[]
    for a,start in corrections:
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert np.all(a['stochastic'][start:]) and not a['ownership'][start:].any()
        assert np.array_equal(np.clip(a['raw_actions'][start:],-1,1).astype(np.float32),a['actions'][start:])
        x=torch.tensor(a['states'],device=device)
        # Label the actual successful sampled action, never the failed mean.
        values.append(dict(x=x,mask=torch.arange(len(x),device=device)>=start,
                           target=torch.tensor(a['raw_actions'],device=device),kind='successful_sample'))
    q=M.frozen(p,p['student_initial']['path']).to(device)
    for a in anchors:
        x=torch.tensor(a['states'],device=device)
        with torch.no_grad():target=q.sequence(x).detach()
        assert np.array_equal(target.clamp(-1,1).cpu().float().numpy(),a['actions'])
        values.append(dict(x=x,mask=torch.ones(len(x),device=device,dtype=torch.bool),
                           target=target,kind='candidate_new_success'))
    assert len(corrections)==len(anchors)==2
    return values


def objective(q,data,refs,p):
    loss=next(q.branch.parameters()).new_zeros(());parts=[]
    for row in data:
        pred=q.sequence(row['x'])[row['mask']]
        value=((pred-row['target'][row['mask']])/pred.new_tensor(p['retention_sigma'])).square().mean()/len(data)
        loss=loss+value;parts.append(float(value.detach()))
    for ref in refs:
        pred=q.sequence(ref['x'])[ref['mask']]
        value=.5*((pred-ref['target'][ref['mask']])/pred.new_tensor(p['retention_sigma'])).square().mean()/len(refs)
        loss=loss+p['keep_weight']*value
    return loss,parts


def fit(obj,p,corrections,anchors,guard,event):
    q,opt=obj;data=dataset(p,corrections,anchors);refs=M.references(q,p)
    body=M.L.fixed(q);trace=[]
    for k in range(p['recovery_steps']):
        guard();before=M.state(obj);opt.zero_grad(set_to_none=True)
        loss,parts=objective(q,data,refs,p);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(M.L.variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        okay=all(torch.isfinite(v).all() for v in M.L.variables(q))
        if not okay:M.load(obj,before)
        trace.append(dict(step=k,loss=float(loss.detach()),parts=parts,
                          gradient_norm=norm,accepted=bool(okay)))
        if not okay:break
    assert M.L.fixed(q)==body
    return dict(attempted_optimizer=len(trace),applied=sum(t['accepted'] for t in trace),trace=trace,
                actual_successful_sample_actions_not_teacher_mean=True,
                full_own_history_BPTT=True,mixed_history_not_onpolicy=True,new_Critic=0)
