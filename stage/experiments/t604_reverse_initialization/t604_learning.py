"""Matched40-history risk aggregation; same frozen-parent recurrent Actor.

Both arms use identical raw-location targets, initialization and Adam steps.
No new RL, teacher query, seeds or architectural change versus T598.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_learning as Old
P,G,torch,np=Old.P,Old.G,Old.torch,Old.np
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest
initial,frozen,variables,fixed,state,load,make,export=Old.initial,Old.frozen,Old.variables,Old.fixed,Old.state,Old.load,Old.make,Old.export


def dataset(p):
    q=initial(p);data=[]
    with torch.no_grad():
        for kind,rows in [('old',p['keep_rows']),('recovery',p['recovery_rows'])]:
            for row in rows:
                assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
                a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);mask=torch.tensor(~a['ownership'].astype(bool))
                assert mask.any()
                base=Old.Parent.Actor.sequence(q,x).detach()
                if kind=='recovery':
                    k=int(a['student_takeover_first']);assert not mask[:k].any() and mask[k:].all()
                    teacher=frozen(p,row['request']['model']);assert teacher.phash()==row['policy_hash']
                    target=teacher.sequence(x).detach()
                    assert float((target-torch.tensor(a['locations'])).abs().max())<1e-12
                    assert np.array_equal(target[k:].clamp(-1,1).float().numpy(),a['actions'][k:])
                else:target=base.clone()
                n=q.normalized(x);reference=q.anchor(n[:,None])[0][:,0].detach()
                data.append(dict(id=row['job_id'],kind=kind,x=x,n=n,base=base,reference=reference,target=target,mask=mask))
    assert len(data)==42 and sum(g['kind']=='old' for g in data)==39
    return data


def objective(q,data,packed,method):
    assert method in ['mean','tail'];pred=q.branch(packed)[0];tasks=[]
    for i,g in enumerate(data):
        loc=(g['base']+(pred[:len(g['n']),i]-g['reference']))[g['mask']]
        per=(loc-g['target'][g['mask']]).square().mean(-1)
        tasks.append(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())
    tasks=torch.stack(tasks);avg=tasks.mean();tail=tasks.topk(10).values.mean()
    loss=avg if method=='mean' else .5*avg+.5*tail
    return loss,dict(mean=float(avg.detach()),worst10=float(tail.detach()),old_mean=float(tasks[:39].mean().detach()),recovery=float(tasks[39:].mean().detach()))


def fit(obj,data,p,guard,event,steps=None):
    q,opt=obj;body=fixed(q);before=Old.measure(q,data);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);trace=[]
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p['method']);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k);trace.append(dict(step=k,losses=parts,gradient_norm=norm))
    assert fixed(q)==body
    return dict(steps=len(trace),trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,method=p['method'],
        trainable_parameters=sum(v.numel() for v in variables(q)),new_RL=0,new_Critic=0,
        raw_location_targets_in_both_groups=True,full_current_residual_history_BPTT=True)


def make(p):
    m=p['student_initial'];assert sha(m['path'])==m['sha256']
    q=Old.frozen(p,m['path']);assert q.phash()==m['policy_hash']
    q.branch.requires_grad_(True)
    return q,torch.optim.Adam(variables(q),lr=p['fit_lr'],foreach=False,fused=False)
