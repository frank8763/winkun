"""One episode-success REINFORCE step, deterministic baseline, fresh SGD.

The stored samples have exactly the initialization's behavior-policy hash.
They are reused historical data, never fresh interaction or demonstrations.
"""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as A
G,P,torch,np=A.G,A.P,A.torch,A.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
def make(p):
    q=A.frozen(p,p['initial']['path']).train().requires_grad_(True)
    opt=torch.optim.SGD(q.parameters(),lr=p['learning_rate'],momentum=0.,foreach=False)
    assert not opt.state and q.phash()==p['initial']['policy_hash'];return q,opt
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()),source_frozen_value_and_original_Adam_reference=obj[0].p['initial'])
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def arrays(p):
    result=[]
    for row in p['training_rows']:
        assert row['policy_hash']==p['initial']['policy_hash'] and sha(row['path'])==row['sha256']
        a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());result.append(a)
    return result
def labels(a):return float(a['end_flags'][-1,1] and not a['end_flags'][:,0].any())
def fit(obj,items,p,guard,event):
    q,opt=obj;guard();opt.zero_grad(set_to_none=True);trace=[];total=0.
    for a in items:
        assert not a['ownership'].any() and a['stochastic'].all() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        x=torch.tensor(a['states']);loc=q.sequence(x);raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
        lp=P.L.density(loc,raw,atom,p['raw_sigma']);me=float((loc.detach()-torch.tensor(a['locations'])).abs().max());le=float((lp.detach()-old).abs().max());assert me<1e-10 and le<1e-9
        outcome=labels(a);adv=outcome-p['deterministic_success_baseline']
        # Sum, not mean, the conditional scores of the whole sampled episode.
        # Exactly one on-policy gradient before any weight changes; no PPO ratio.
        loss=-adv*lp.sum()/len(items);loss.backward();total+=float(loss.detach())
        trace.append(dict(success=outcome,advantage=adv,decisions=len(x),density_error=[me,le],log_likelihood_sum=float(lp.detach().sum())))
    norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));assert norm>0.
    event('BEFORE',0);opt.step();event('AFTER',0);assert not opt.state
    return dict(Actor_SGD=1,Actor_Adam=0,Value_Adam=0,new_policy_gradient_steps=1,trajectory_success_objective=True,trace=trace,loss=total,gradient_norm_before_clip=norm,whole_actor_parameters=sum(z.numel() for z in q.parameters()),historical_reuse_NOT_independent_new_data=True)
def export(obj,p,path):
    s=state(obj);b=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),architecture='T501_single120_64carry_two_neural_command_residuals',new_optimizer='SGD_NO_MOMENTUM',source_immutable_Value_and_historical_Adam=p['initial'],single_actor=True),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];Path(path).write_bytes(blob)
    return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=obj[0].phash(),joint_hash=digest(s))
def trust(p,q,items):
    original=A.frozen(p,p['initial']['path']);values=[]
    with torch.no_grad():
        for a in items:
            x=torch.tensor(a['states']);old=original.sequence(x);new=q.sequence(x)
            # Upper bound on float32-clipped action KL, on fixed recorded
            # histories only. This is not a closed-loop safety certificate.
            kl=.5*((new-old)/new.new_tensor(p['raw_sigma'])).square().sum(-1)
            values.append(dict(mean=float(kl.mean()),max=float(kl.max())))
    return dict(rows=values,passed=all(v['mean']<=p['max_mean_KL'] and v['max']<=p['max_point_KL'] for v in values))
