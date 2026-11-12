"""Fixed successful-history likelihood replay, explicitly off-policy supervision.

No new policy-gradient samples, teacher, learned value or runtime intervention.
Complete recurrent histories are retained; only frozen-backbone features cache.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'))
import t501_learning as A
G,P,torch,np=A.G,A.P,A.torch,A.np
read,sha,digest=G.read,G.sha,G.digest

def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=P.checkpoint(p['initial']['path'])['joint_state'];q=A.Actor(p);q.load_state_dict(old['actor']);assert q.phash()==p['initial']['policy_hash']
    q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(A.variables(q),lr=p['actor_lr'],foreach=False,fused=False);ao.load_state_dict(old['actor_opt'])
    for group in ao.param_groups:group['lr']=p['actor_lr']
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    for params,key in [(q.drive_adapter.parameters(),'drive_initial_steps'),(q.steer_adapter.parameters(),'steer_initial_steps')]:
        assert all(int(ao.state[z]['step'])==p[key] for z in params)
    if p.get('initial_joint_locked'):assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    return q,v,ao,vo

def cached(q,a):
    x=torch.tensor(a['states']);n=q.normalized(x)
    with torch.no_grad():
        h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        return dict(features=torch.cat([h,n],-1),logits=q.head(h),target=q.location(h,n))

def references(p):
    q=make(p)[0].requires_grad_(False);refs=[]
    for row in p['keep_rows']:
        assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
        a=G.E.N.S.arrays(row['path']);g=cached(q,a);key=row['job_id']
        family=3 if key.startswith('t502_') else 2 if key.startswith('wide117_8m_') else 1 if key.startswith('wide117_80_') else 0
        g.update(id=key,family=family,source_sha256=row['sha256']);refs.append(g)
    assert [sum(g['family']==i for g in refs) for i in range(4)]==[271,54,54,6]
    return refs

def training_data(p):
    q=make(p)[0].requires_grad_(False);data=[]
    for row in p['replay_rows']:
        assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
        a=G.E.N.S.arrays(row['path']);assert not a['ownership'].any() and a['stochastic'].all()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist()==[0,1,0]
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        g=cached(q,a);goal=a['task_goal_xy'].reshape(-1,2)[0]
        distance=np.linalg.norm(a['poses'][:,:2]-goal,axis=1);near=distance<=p['near_goal_radius_m'];assert near.any()
        g.update(id=row['job_id'],source_sha256=row['sha256'],raw=torch.tensor(a['raw_actions']),atom=torch.tensor(a['atom']),near=torch.tensor(near),distance=distance,dt=torch.tensor(a['native_counts'].astype(float)))
        data.append(g)
    assert len(data)==2 and len({g['id'] for g in data})==1
    return data

def preserve(q,refs,p):
    counts=[sum(g['family']==i for g in refs) for i in range(4)];total=0.
    for g in refs:
        loc=A.location(q,g);err=.5*((loc-g['target'])/loc.new_tensor(p['raw_sigma'])).square().mean(-1)
        loss=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/(4*counts[g['family']])
        (p['keep_weight']*loss).backward();total+=float(loss.detach())
    return total

def replay_loss(q,g,p):
    loc=A.location(q,g);nll=-P.L.density(loc,g['raw'],g['atom'],p['raw_sigma'])
    w=g['dt'];near=g['near']
    return .5*(nll*w).sum()/w.sum()+.5*(nll[near]*w[near]).sum()/w[near].sum()

def fit(obj,data,p,refs,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=A.fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());trace=[]
    with torch.no_grad():initial=sum(float(replay_loss(q,g,p)) for g in data)/len(data)
    for k in range(p['epochs'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=0.
        for g in data:
            loss=replay_loss(q,g,p)/len(data);loss.backward();total+=float(loss.detach())
        keep=preserve(q,refs,p);norm=float(torch.nn.utils.clip_grad_norm_(A.variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();event('AFTER',k)
        with torch.no_grad():
            bound=sum(float((.5*((A.location(q,g)-g['target'])/g['target'].new_tensor(p['raw_sigma'])).square().sum(-1)*g['dt']).sum()/g['dt'].sum()) for g in data)/len(data)
        trace.append(dict(epoch=k,successful_replay_NLL=total,successful_history_risk_keep=keep,gradient_norm=norm,initial_history_rawGaussian_KL_bound=bound))
        assert A.fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        if bound>p['target_kl']:break
    with torch.no_grad():final=sum(float(replay_loss(q,g,p)) for g in data)/len(data)
    n=len(trace)
    assert all(int(ao.state[z]['step'])==p['drive_initial_steps']+n for z in q.drive_adapter.parameters())
    assert all(int(ao.state[z]['step'])==p['steer_initial_steps']+n for z in q.steer_adapter.parameters())
    return dict(Actor_Adam=n,Value_Adam=0,new_policy_gradient_updates=0,trace=trace,initial_replay_NLL=initial,final_replay_NLL=final,trainable_parameters=11906,supervised_offpolicy_success_likelihood_NOT_original_SIL_or_AWR=True)
