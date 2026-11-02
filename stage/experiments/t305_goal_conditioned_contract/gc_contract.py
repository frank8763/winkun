"""Goal-conditioned learning primitives; no native/optimizer/policy update.

Finite-horizon discounted decision-endpoint occupancy with explicit terminal
tokens. This is not calibrated safe first-hit Q or an exact reproduction of
the continuing-MDP CRL theorem. The original Actor distribution is unmodified.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t295_hindsight_support'))
import t295_contract as H
sys.path.insert(0,str(ROOT/'stage/experiments/t296_log_tail_support'))
import stable_density as D
sys.path.insert(0,str(ROOT/'stage/experiments/t302_balanced_bridge_learning'))
import t302_learning as T
np,torch,C,S,R=H.np,H.torch,H.C,H.S,H.R


def relative(pose,world):
    d=np.asarray(world,np.float64)-np.asarray(pose[:2],np.float64)
    c,s=math.cos(float(pose[2])),math.sin(float(pose[2]))
    return np.stack([c*d[...,0]+s*d[...,1],-s*d[...,0]+c*d[...,1]],axis=-1)


def query_matrix(poses,world_queries):
    return np.stack([relative(p,world_queries) for p in poses])


def future_law(counts,flags,start,cap,complete,gamma=R.GAMMA):
    counts=np.asarray(counts,np.int64);flags=np.asarray(flags,bool)
    assert counts.ndim==1 and flags.shape==(len(counts),3)
    assert np.all((counts>0)&(counts<=100)) and 0<=start<len(counts)
    assert not flags[:-1].any() and int(counts.sum())<=cap
    survival=1.;records=[]
    for j in range(start,len(counts)):
        contact,goal,timeout=flags[j]
        # Collision takes precedence even for a synthetic contact+goal overlap.
        if contact:
            records.append(dict(index=j,type='collision',mass=survival));survival=0.;break
        duration_discount=gamma**int(counts[j]);mass=survival*(1-duration_discount)
        records.append(dict(index=j,type='coordinate',mass=mass));survival*=duration_discount
        if goal:
            records.append(dict(index=j,type='coordinate_absorbed_goal',mass=survival));survival=0.;break
        if timeout:
            assert complete and int(counts.sum())==cap,'Timeout token requires true task cap'
            records.append(dict(index=j,type='task_timeout',mass=survival));survival=0.;break
    if survival:
        assert not complete,'A complete episode needs native termination flags'
        records.append(dict(index=-1,type='collector_censored_unknown',mass=survival))
    assert abs(sum(r['mass'] for r in records)-1)<2e-12
    return records


def draw(law,uniform):
    assert 0<=uniform<1
    total=0.
    for record in law:
        total+=record['mass']
        if uniform<total:return dict(record)
    assert abs(total-1)<2e-12
    return dict(law[-1])


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_action=torch.nn.Sequential(torch.nn.Linear(124,64),torch.nn.Tanh(),torch.nn.Linear(64,32))
        self.query=torch.nn.Sequential(torch.nn.Linear(6,32),torch.nn.Tanh(),torch.nn.Linear(32,32))

    def logits(self,behavior120,previous_latent,actions,body_queries,types):
        assert body_queries.ndim==3 and body_queries.shape[:2]==types.shape
        x=torch.cat([behavior120,previous_latent,actions],-1)
        onehot=torch.nn.functional.one_hot(types,num_classes=4).to(x.dtype)
        g=torch.cat([body_queries,onehot],-1)
        phi=self.state_action(x);psi=self.query(g)
        return (phi[:,None,:]*psi).sum(-1)


def contrastive_objective(logits,valid):
    assert logits.ndim==2 and logits.shape[0]==logits.shape[1]
    assert valid.dtype==torch.bool and valid.shape==(len(logits),)
    # Unknown collector-censor tokens are neither positives nor negatives.
    kept=torch.nonzero(valid,as_tuple=False)[:,0];assert len(kept)>1
    z=logits[kept][:,kept]
    nce=torch.nn.functional.cross_entropy(z,torch.arange(len(z)))
    reg=.01*torch.logsumexp(z,dim=1).square().mean()
    return nce+reg


def conditional_action(mean,previous_mean,previous_action,reset,innovation):
    previous,_=D.terms(previous_mean,previous_action)
    rho=torch.where(reset,torch.zeros_like(reset,dtype=torch.float64),torch.full_like(reset,D.RHO,dtype=torch.float64))[:,None]
    latent=rho*previous+torch.sqrt(1-rho.square())*innovation
    return T.L.AR.sample(mean,latent,native=False),previous


def proposed_actor_objective(q,critic,behavior120,query120,previous_query120,
                             previous_action,reset,innovation,body_goal):
    """Differentiable interface only, never authorized by random critic scores.

The critic retains behavior-goal context; the Actor uses fully reconstructed
query-goal input. Transfer from behavior continuation to a different query policy
requires real control validation, not the ordinary continuing-MDP Q identity.
"""
    mean=torch.cat([q.step(row[None])[1] for row in query120])
    previous_mean=torch.cat([q.step(row[None])[1] for row in previous_query120])
    action,prev_latent=conditional_action(mean,previous_mean,previous_action,reset,innovation)
    kind=torch.zeros((len(mean),1),dtype=torch.long)
    value=critic.logits(behavior120,prev_latent.to(behavior120.dtype),action.to(behavior120.dtype),body_goal[:,None],kind)[:,0]
    return -value.mean(),action
