"""All-outcome empirical MLE; no new on-policy interactions or teacher labels."""
from pathlib import Path
import sys,copy,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t717_onpolicy_recovery'));import t717_learning as Base
A,B,torch,np,digest=Base.A,Base.B,Base.torch,Base.np,Base.digest
DENSITY=Base.DENSITY

def make(p):
    torch.set_num_threads(p['fit_threads']);torch.use_deterministic_algorithms(True)
    q=A.frozen(p,p['learner_initial']['path']);q.requires_grad_(False);q.replay_head.requires_grad_(True);assert sum(x.numel() for x in q.parameters() if x.requires_grad)==20198
    opt=torch.optim.Adam(q.replay_head.parameters(),lr=p['actor_lr'],foreach=False,fused=False);return q,opt

def state(obj):return copy.deepcopy(dict(actor=obj[0].state_dict(),actor_opt=obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(copy.deepcopy(s['actor']));obj[1].load_state_dict(copy.deepcopy(s['actor_opt']))
def dataset(p,q,guard):
    groups={key:[] for key in p['training_ids']};manifest=[]
    for row in p['replay_rows']:
        guard();assert B.sha(row['path'])==row['sha256'];arr=A.Old.Existing.L.T.R.E.N.S.arrays(row['path']);k=int(arr['recovery_first']);assert k>0 and len(arr['states'])>k and arr['stochastic'][k:].all() and not arr['stochastic'][:k].any();assert arr['ownership'][:k].all() and not arr['ownership'][k:].any()
        assert row['job_id'] in groups and not row.get('exact',False)
        features=Base.features(q,arr);score=1 if row['success'] and not row['collision'] and not ('fullfield_start_pose' in row['request']['job'] and row.get('field_boundary')) else -1 if row['collision'] else 0
        g=dict(id=row['job_id'],f=features['f'][k:].detach(),base=features['base'][k:].detach(),raw=torch.tensor(arr['raw_actions'][k:]),atom=torch.tensor(arr['atom'][k:]),score=score,path=row['path'],first=k)
        groups[g['id']].append(g);manifest.append(dict(path=row['path'],sha256=row['sha256'],id=g['id'],score=score,first=k,suffix=len(g['f'])))
    for key,items in groups.items():
        assert len(items)==6;den=sum(math.exp(g['score']/p['weight_beta']) for g in items)
        for g in items:g['weight']=math.exp(g['score']/p['weight_beta'])/den/len(groups);assert g['weight']>0
    assert len(groups)==12 and sum(len(v) for v in groups.values())==72
    assert abs(sum(g['weight'] for items in groups.values() for g in items)-1)<1e-12
    return groups,manifest

def fit(obj,data,p,guard,event,steps=None):
    q,opt=obj;fixed=Base.fixed(q);trace=[]
    for epoch in range(p['fit_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);total=0.
        for items in data.values():
            for g in items:
                lp=DENSITY(Base.location(q,g),g['raw'],g['atom'],p['raw_sigma']);loss=-g['weight']*lp.mean();loss.backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.replay_head.parameters(),.5,error_if_nonfinite=True));event('BEFORE',epoch);opt.step();event('AFTER',epoch)
        assert Base.fixed(q)==fixed and all(torch.isfinite(x).all() for x in q.parameters());trace.append(dict(epoch=epoch,weighted_NLL=total,gradient_norm=norm))
    return dict(trace=trace,Actor_Adam=len(trace),trainable_parameters=20198,all72_included=True,strictly_positive_weights=True,equal_ID_and_pertrajectory_normalization=True,no_Value_or_ppo=True,frozen_body=fixed)

def export(q,p,path):
    import io,gzip
    joint=dict(actor=copy.deepcopy(q.state_dict()));b=io.BytesIO();torch.save(dict(joint_state=joint,architecture=B.ARCH,policy_hash=q.phash(),joint_hash=digest(joint),new_RL=False,offpolicy_outcome_weighted_MLE=True,optimizer_separate=True,runtime_teacher=False),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<8*2**20
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=B.sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(joint))
