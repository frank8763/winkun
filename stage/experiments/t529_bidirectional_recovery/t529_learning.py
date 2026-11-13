"""Bidirectional real recovery and successful sequence rehearsal, no fake PPO."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t523_joint_failure_signal'));import t523_learning as Base
G,P,A,torch,np,digest,sha=Base.G,Base.P,Base.A,Base.torch,Base.np,Base.digest,Base.sha
make=Base.make;variables=Base.variables

def references(p,own_row):
    refs=[]
    for row in p['keep_rows']+[own_row]:
        assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        source=A.frozen(p,row['request']['model']);x=torch.tensor(a['states']);target=torch.tensor(a['locations'])
        with torch.no_grad():assert source.phash()==row['policy_hash'] and float((source.sequence(x)-target).abs().max())<1e-10
        family=2 if row is own_row else int(row['job_id'].startswith('t502_'));refs.append(dict(x=x,target=target,id=row['job_id'],family=family,source_policy_hash=row['policy_hash']))
    assert [sum(g['family']==i for g in refs) for i in range(3)]==[8,11,1];return refs

def preserve(q,refs):
    total=0.;counts=[8,11,1]
    for g in refs:
        loc=q.sequence(g['x']);err=.5*((loc-g['target'])/loc.new_tensor(q.p['raw_sigma'])).square().mean(-1)
        loss=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/(3*counts[g['family']]);loss.backward();total+=float(loss.detach())
    return total

def fit(obj,a,p,refs,guard,event,steps=None):
    q,v,ao,vo=obj;vh=digest(v.state_dict());voh=digest(vo.state_dict());x=torch.tensor(a['states']);owner=torch.tensor(a['ownership'].astype(bool));target=torch.tensor(a['teacher_actions'],dtype=torch.float64);assert owner.any() and not a['stochastic'].any();trace=[]
    previous=[int(ao.state[z]['step']) for z in variables(q)]
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);loc=q.sequence(x);err=P.action_loss(loc[owner],target[owner]).mean(-1);recovery=.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean();recovery.backward();keep=preserve(q,refs);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        assert digest(v.state_dict())==vh and digest(vo.state_dict())==voh;trace.append(dict(epoch=k,old_recovery_action_loss=float(recovery.detach()),old8_field11_and_newgoal_keep=keep,gradient_norm=norm))
    n=len(trace);assert [int(ao.state[z]['step']) for z in variables(q)]==[s+n for s in previous]
    return dict(Actor_Adam=n,Value_Adam=0,new_policy_gradient_updates=0,trace=trace,all_three_success_families_replayed=True,only_verified_teacher_owned_labels=True,full_history_BPTT=True)
