"""Validated recovery imitation and actual successful-history rehearsal.

This bounded first correction pilot has no new policy-gradient/Value updates.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t523_joint_failure_signal'))
import t523_learning as Base
G,P,A,torch,np,digest,sha=Base.G,Base.P,Base.A,Base.torch,Base.np,Base.digest,Base.sha
make=Base.make
variables=Base.variables

def references(p):
    q=make(p)[0].requires_grad_(False);refs=[]
    for row in p['keep_rows']:
        assert row['policy_hash']==p['initial']['policy_hash'] and row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
        a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        x=torch.tensor(a['states']);target=torch.tensor(a['locations'])
        with torch.no_grad():assert float((q.sequence(x)-target).abs().max())<1e-10
        refs.append(dict(x=x,target=target,id=row['job_id'],family=int(row['job_id'].startswith('t502_'))))
    assert [sum(x['family']==i for x in refs) for i in [0,1]]==[8,11]
    return refs

def fit(obj,a,p,refs,guard,event,steps=None):
    q,v,ao,vo=obj;vh=digest(v.state_dict());voh=digest(vo.state_dict());x=torch.tensor(a['states']);owner=torch.tensor(a['ownership'].astype(bool));y=torch.tensor(a['teacher_actions'],dtype=torch.float64)
    assert owner.any() and not a['stochastic'].any();trace=[];before=[int(ao.state[z]['step']) if z in ao.state else 0 for z in variables(q)]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);loc=q.sequence(x)
        err=P.action_loss(loc[owner],y[owner]).mean(-1);recovery=.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean();recovery.backward()
        keep=Base.preserve(q,refs,p);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        assert digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        trace.append(dict(epoch=k,recovery_action_loss=float(recovery.detach()),actual_success_history_keep=keep,gradient_norm=norm))
    n=len(trace);assert [int(ao.state[z]['step']) for z in variables(q)]==[b+n for b in before]
    return dict(Actor_Adam=n,Value_Adam=0,new_policy_gradient_updates=0,trace=trace,teacher_owned_labels_only=True,actual_same_initial_success_history_targets=True,full_history_BPTT=True)
