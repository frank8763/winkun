"""Task-balanced recovery samples and verified nominal eight-metre behavior."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_learning as Previous
M=Previous.M
torch,np=M.torch,M.np


def dataset(p,load,device='cuda'):
    values=[]
    rows=p['field_recovery_rows'];assert len(rows)==3
    for row in rows:
        a=load(row);start=row['request']['job']['student_takeover_first']
        assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        assert a['stochastic'][start:].all() and not a['ownership'][start:].any()
        assert np.array_equal(np.clip(a['raw_actions'][start:],-1,1).astype(np.float32),a['actions'][start:])
        x=torch.tensor(a['states'],device=device)
        values.append(dict(x=x,mask=torch.arange(len(x),device=device)>=start,
                           target=torch.tensor(a['raw_actions'],device=device),
                           weight=.25/len(rows),kind='field_successful_sample'))
    for row,model,weight,kind in [(p['nominal_eight_row'],p['reference_model'],.25,'nominal_eight_success')]+[
            (r,p['student_initial'],.25,'new_success_anchor') for r in p['new_anchor_rows']]:
        a=load(row);assert not a['ownership'].any() and not a['stochastic'].any()
        q=M.frozen(p,model['path']).to(device);x=torch.tensor(a['states'],device=device)
        with torch.no_grad():target=q.sequence(x).detach()
        assert np.array_equal(target.clamp(-1,1).cpu().float().numpy(),a['actions'])
        values.append(dict(x=x,mask=torch.ones(len(x),dtype=torch.bool,device=device),
                           target=target,weight=weight,kind=kind))
    assert len(values)==6 and abs(sum(v['weight'] for v in values)-1)<1e-12
    return values


def objective(q,data,refs,p):
    loss=next(q.branch.parameters()).new_zeros(());terms=[]
    for row in data:
        pred=q.sequence(row['x'])[row['mask']]
        value=((pred-row['target'][row['mask']])/pred.new_tensor(p['retention_sigma'])).square().mean()*row['weight']
        loss=loss+value;terms.append(float(value.detach()))
    for ref in refs:
        pred=q.sequence(ref['x'])[ref['mask']]
        value=.5*((pred-ref['target'][ref['mask']])/pred.new_tensor(p['retention_sigma'])).square().mean()/len(refs)
        loss=loss+p['keep_weight']*value
    return loss,terms


def fit(obj,p,data,guard,event):
    q,opt=obj;refs=M.references(q,p);body=M.L.fixed(q);trace=[]
    for k in range(p['recovery_steps']):
        guard();before=M.state(obj);opt.zero_grad(set_to_none=True)
        loss,terms=objective(q,data,refs,p);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(M.L.variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        okay=all(torch.isfinite(v).all() for v in M.L.variables(q))
        if not okay:M.load(obj,before)
        trace.append(dict(step=k,loss=float(loss.detach()),terms=terms,
                          gradient_norm=norm,accepted=bool(okay)))
        if not okay:break
    assert M.L.fixed(q)==body
    return dict(attempted_optimizer=len(trace),applied=sum(t['accepted'] for t in trace),
                trace=trace,task_weight=[.25,.25,.25,.25],field_trajectory_weights=[.25/3]*3,
                actual_successful_actions_only=True,nominal_eight_not_recovery_label=True,
                full_own_history_BPTT=True,new_Critic=0,new_RL=0)
