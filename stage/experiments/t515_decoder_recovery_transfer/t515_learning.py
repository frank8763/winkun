"""Only the existing 130-parameter action decoder learns actual teacher suffix."""
from pathlib import Path
import copy,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t504_success_replay'));import t504_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest

def variables(q):return list(q.head.parameters())
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('head.')})
def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    saved=P.checkpoint(p['initial']['path']);old=saved['joint_state'];q=A.Actor(p)
    q.load_state_dict(old['actor']);assert q.phash()==p['initial']['policy_hash']
    q.requires_grad_(False);q.head.requires_grad_(True)
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(A.variables(q),lr=p['actor_lr'],foreach=False,fused=False)
    registered=len(old['actor_opt']['param_groups'])==2
    if not registered:ao.load_state_dict(old['actor_opt'])
    ao.add_param_group(dict(params=variables(q),lr=p['decoder_lr']))
    if registered:ao.load_state_dict(old['actor_opt'])
    assert len(ao.param_groups)==2 and sum(z.numel() for z in variables(q))==130
    assert all(not z.requires_grad and int(ao.state[z]['step'])==p['drive_initial_steps'] for z in q.drive_adapter.parameters())
    assert all(not z.requires_grad and int(ao.state[z]['step'])==p['steer_initial_steps'] for z in q.steer_adapter.parameters())
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    if p.get('initial_joint_locked'):assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    return q,v,ao,vo

def location(q,g):
    f=g['features'];res=torch.stack([q.drive_adapter(f)[:,0],q.steer_adapter(f)[:,0]],-1)
    return 1.25*torch.tanh(q.head(f[:,:64])+.3*torch.tanh(res))

def groups(p):
    q=make(p)[0].requires_grad_(False);data=[]
    for i,row in enumerate(p['keep_rows']):
        assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
        a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        assert not a['ownership'].any() and not a['stochastic'].any()
        g=B.cached(q,a);assert float((g['target']-torch.tensor(a['locations'])).abs().max())<1e-10
        g.update(kind='keep',family=int(row['job_id'].startswith('t502_')),id=row['job_id'],record_id='keep_'+str(i),source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'].astype(float)))
        data.append(g)
    row=p['repair_rows'][0];assert row['teacher'] and row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
    a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    g=B.cached(q,a);assert float((g['target']-torch.tensor(a['locations'])).abs().max())<1e-10
    k=p['teacher_first'];own=torch.tensor(a['ownership'],dtype=torch.bool)
    assert not own[:k].any() and own[k:].all() and not a['stochastic'].any()
    prefix={key:value[:k].clone() for key,value in g.items()}
    prefix.update(kind='prefix',family=0,id=row['job_id'],record_id='same_task_prefix',source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'][:k].astype(float)))
    data.append(prefix)
    repair={key:value[own].clone() for key,value in g.items()};repair['target']=torch.tensor(a['teacher_actions'][a['ownership'].astype(bool)],dtype=torch.float64)
    repair.update(kind='repair',family=0,id=row['job_id'],record_id='repair_0',source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'][a['ownership'].astype(bool)].astype(float)))
    data.append(repair);assert len(data)==21
    return data

def loss(q,g,p):
    loc=location(q,g);sigma=loc.new_tensor(p['raw_sigma'])
    axis=.5*P.action_loss(loc,g['target'])/sigma.square() if g['kind']=='repair' else .5*((loc-g['target'])/sigma).square()
    err=axis.mean(-1);dt=g['dt']
    return .5*(err*dt).sum()/dt.sum()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()
def measure(q,data,p):
    with torch.no_grad():return {g['record_id']:float(loss(q,g,p)) for g in data}

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict())
    oldopt=digest({i:ao.state_dict()['state'][i] for i in range(8)});before=[int(ao.state[z]['step']) if z in ao.state else 0 for z in variables(q)]
    initial=measure(q,data,p);trace=[]
    for k in range(p['block_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=dict(repair=0.,keep=0.,prefix=0.)
        for g in data:
            if g['kind']=='repair':weight=.5
            elif g['kind']=='prefix':weight=.1
            else:weight=.4/(2*sum(h['kind']=='keep' and h['family']==g['family'] for h in data))
            z=loss(q,g,p);(weight*z).backward();total[g['kind']]+=weight*float(z.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(step=k,repair=total['repair'],keep=total['keep'],prefix=total['prefix'],gradient_norm=norm))
        assert fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
        assert digest({i:ao.state_dict()['state'][i] for i in range(8)})==oldopt
    assert [int(ao.state[z]['step']) for z in variables(q)]==[s+len(trace) for s in before]
    return dict(Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=measure(q,data,p),trace=trace,trainable_parameters=130,old_adapter_Adam_unchanged=True)
