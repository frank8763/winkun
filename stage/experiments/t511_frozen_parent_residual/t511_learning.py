"""Frozen accepted parent plus a zero-initialized globally causal residual.

Only the new 11970-parameter head has a fresh optimizer. Original parent
parameters and its mature Adam remain immutable in the protected parent file.
"""
from pathlib import Path
import gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t509_two_goal_joint'));import t509_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest
class Actor(A.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['residual_seed']);self.goal_residual=torch.nn.Sequential(torch.nn.Linear(184,64),torch.nn.Tanh(),torch.nn.Linear(64,2)).double()
            torch.nn.init.zeros_(self.goal_residual[-1].weight);torch.nn.init.zeros_(self.goal_residual[-1].bias)
    def location(self,h,n):
        return (super().location(h,n)+.1*torch.tanh(self.goal_residual(torch.cat([h,n],-1)))).clamp(-1.25,1.25)
    def phash(self):return digest(dict(architecture='T511_single120_64carry_frozen_parent_global_residual',state=self.state_dict(),sigma=self.p['raw_sigma']))
def variables(q):return list(q.goal_residual.parameters())
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('goal_residual.')})
def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);old=P.checkpoint(p['initial']['path'])['joint_state'];q=Actor(p)
    missing,unexpected=q.load_state_dict(old['actor'],strict=False);assert not unexpected
    expanded=any(k.startswith('goal_residual.') for k in old['actor']);assert not missing if expanded else sorted(missing)==sorted('goal_residual.'+k for k in q.goal_residual.state_dict())
    q.requires_grad_(False);q.goal_residual.requires_grad_(True);assert sum(z.numel() for z in variables(q))==11970
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False);ao=torch.optim.Adam(variables(q),lr=p['actor_lr'],foreach=False,fused=False)
    if expanded:ao.load_state_dict(old['actor_opt'])
    else:assert not ao.state
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    if p.get('initial_joint_locked'):assert q.phash()==p['initial']['policy_hash'] and digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    return q,v,ao,vo
def frozen(p,path):
    d=P.checkpoint(path)
    if not any(k.startswith('goal_residual.') for k in d['joint_state']['actor']):return A.frozen(p,path)
    q=Actor(p).eval().requires_grad_(False);q.load_state_dict(d['joint_state']['actor']);assert q.phash()==d['policy_hash'];return q
def export(obj,path,limit):
    s=P.state(*obj);buf=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),single_actor=True,teacher_at_deployment=False,architecture='T511_single120_64carry_frozen_parent_global_residual'),buf);b=gzip.compress(buf.getvalue(),mtime=0);assert len(b)<=limit;Path(path).write_bytes(b)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(s),bytes=len(b))
def location(q,g):return (A.location(q,g)+.1*torch.tanh(q.goal_residual(g['features']))).clamp(-1.25,1.25)
def groups(p):
    q=make(p)[0].requires_grad_(False);data=[]
    for kind,key in [('repair','repair_rows'),('keep','keep_rows')]:
        for i,row in enumerate(p[key]):
            assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items());assert not a['ownership'].any()
            g=B.B.cached(q,a);assert float((g['target']-torch.tensor(a['locations'])).abs().max())<1e-10
            if kind=='repair':
                assert a['stochastic'].all();raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);assert float((P.L.density(g['target'],raw,atom,p['raw_sigma'])-torch.tensor(a['logp'])).abs().max())<1e-9
                target=torch.tensor(a['actions'],dtype=torch.float64);near=np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'].reshape(-1,2)[0],axis=1)<=1.;assert near.any();g['near']=torch.tensor(near)
            else:assert not a['stochastic'].any();target=torch.tensor(a['locations'])
            g.update(kind=kind,family=p['new_task_ids'].index(row['job_id']) if kind=='repair' else int(row['job_id'].startswith('t502_')),id=row['job_id'],record_id=kind+'_'+str(i),target=target,source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'].astype(float)));data.append(g)
    assert len(data)==27;return data
def loss(q,g,p):
    loc=location(q,g);sigma=loc.new_tensor(p['raw_sigma']);axis=.5*P.action_loss(loc,g['target'])/sigma.square() if g['kind']=='repair' else .5*((loc-g['target'])/sigma).square();err=axis.mean(-1);dt=g['dt']
    if g['kind']=='repair':n=g['near'];return .5*(err*dt).sum()/dt.sum()+.5*(err[n]*dt[n]).sum()/dt[n].sum()
    return .5*(err*dt).sum()/dt.sum()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()
def measure(q,data,p):
    with torch.no_grad():return {g['record_id']:float(loss(q,g,p)) for g in data}
def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());initial=measure(q,data,p);trace=[];before=[int(ao.state[z]['step']) if z in ao.state else 0 for z in variables(q)]
    for k in range(p['block_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=dict(repair=0.,keep=0.)
        for g in data:
            n=sum(h['kind']==g['kind'] and h['family']==g['family'] for h in data);z=loss(q,g,p)/(2*n);(.5*z).backward();total[g['kind']]+=float(z.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k);trace.append(dict(step=k,repair=total['repair'],keep=total['keep'],norm=norm))
        assert fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    assert [int(ao.state[z]['step']) for z in variables(q)]==[s+len(trace) for s in before]
    return dict(Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=measure(q,data,p),trace=trace,trainable_parameters=11970,frozen_parent_hash=fixed)
