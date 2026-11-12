"""Joint neural drive/steer repair, frozen backbone, actual success retention.

No runtime task selector or teacher. Old longitudinal Adam moments are mapped
explicitly; newly added steering parameters begin with no optimizer history.
"""
from pathlib import Path
import copy,gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'))
import speed_actor as S
G,P,torch,np=S.G,S.P,S.torch,S.np
read,sha,digest=G.read,G.sha,G.digest
LEGACY_FROZEN=S.frozen

class Actor(S.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['steer_seed']);self.steer_adapter=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).double()
            torch.nn.init.zeros_(self.steer_adapter[-1].weight);torch.nn.init.zeros_(self.steer_adapter[-1].bias)
    def location(self,h,n):
        logits=self.head(h).clone();features=torch.cat([h,n],-1)
        logits[...,0]=logits[...,0]+.3*torch.tanh(self.drive_adapter(features)[...,0])
        logits[...,1]=logits[...,1]+.3*torch.tanh(self.steer_adapter(features)[...,0])
        return 1.25*torch.tanh(logits)
    def phash(self):return digest(dict(architecture='T501_single120_64carry_two_neural_command_residuals',state=self.state_dict(),sigma=self.p['raw_sigma']))

def variables(q):return list(q.drive_adapter.parameters())+list(q.steer_adapter.parameters())
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith(('drive_adapter.','steer_adapter.'))})
def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    saved=P.checkpoint(p['initial']['path']);old=saved['joint_state'];q=Actor(p)
    missing,unexpected=q.load_state_dict(old['actor'],strict=False);assert sorted(missing)==sorted('steer_adapter.'+k for k in q.steer_adapter.state_dict()) and not unexpected
    q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==11906
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(variables(q),lr=.001,foreach=False,fused=False)
    state=copy.deepcopy(old['actor_opt']);assert len(state['param_groups'])==1 and state['param_groups'][0]['params']==list(range(4)) and set(state['state'])==set(range(4))
    state['param_groups'][0]['params']=list(range(8));ao.load_state_dict(state)
    assert len(ao.state)==4
    for param,i in zip(q.drive_adapter.parameters(),range(4)):
        assert int(ao.state[param]['step'])==640 and digest(ao.state[param])==digest(old['actor_opt']['state'][i])
    assert all(param not in ao.state for param in q.steer_adapter.parameters())
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    assert digest(v.state_dict())==digest(old['value']) and digest(vo.state_dict())==digest(old['value_opt'])
    return q,v,ao,vo

def frozen(p,path):
    state=P.checkpoint(path)
    if not any(k.startswith('steer_adapter.') for k in state['joint_state']['actor']):return LEGACY_FROZEN(p,path)
    q=Actor(p).eval().requires_grad_(False);q.load_state_dict(state['joint_state']['actor']);assert q.phash()==state['policy_hash'];return q

def export(obj,path,limit):
    state=P.state(*obj);buf=io.BytesIO();torch.save(dict(joint_state=state,policy_hash=obj[0].phash(),joint_hash=digest(state),architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,teacher_at_deployment=False),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=limit;Path(path).write_bytes(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(state),bytes=len(blob))

def classify(row,kind):
    key=row['job_id']
    if kind=='repair':
        return 2 if key.startswith('wide117_80_') else 0 if 'task4_' in key else 1
    return 2 if key.startswith('wide117_8m_') else 1 if key.startswith('wide117_80_') else 0

def groups(p):
    q=make(p)[0].requires_grad_(False);data=[]
    with torch.no_grad():
        for kind,rows in [('repair',p['repair_rows']),('keep',p['keep_rows'])]:
            for r in rows:
                assert r['success'] and not r['collision'] and sha(r['path'])==r['sha256']
                a=G.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
                x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
                target=torch.tensor(a['actions'] if kind=='repair' else a['locations'],dtype=torch.float64)
                loc=q.location(h,n)
                if kind=='keep':assert float((loc-target).abs().max())<1e-10
                data.append(dict(kind=kind,family=classify(r,kind),id=r['job_id'],features=torch.cat([h,n],-1),logits=q.head(h),target=target,source_sha256=r['sha256']))
    assert len(data)==383 and len({g['id'] for g in data})==383
    assert [sum(g['kind']=='repair' and g['family']==i for g in data) for i in range(3)]==[7,35,1]
    assert [sum(g['kind']=='keep' and g['family']==i for g in data) for i in range(3)]==[233,53,54]
    return data

def location(q,g):
    f=g['features'];res=torch.stack([q.drive_adapter(f)[:,0],q.steer_adapter(f)[:,0]],dim=-1)
    return 1.25*torch.tanh(g['logits']+.3*torch.tanh(res))

def measure(q,data):
    values={kind:[[] for _ in range(3)] for kind in ['repair','keep']}
    with torch.no_grad():
        for g in data:
            loc=location(q,g);err=P.action_loss(loc,g['target']) if g['kind']=='repair' else (loc-g['target']).square()
            values[g['kind']][g['family']].append(float(err.mean()))
    return {kind:dict(family_mean_MSE=[sum(v)/len(v) for v in rows],max_episode_MSE=max(max(v) for v in rows)) for kind,rows in values.items()}

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());initial=measure(q,data);assert initial['keep']['max_episode_MSE']<1e-20
    counts={kind:[sum(g['kind']==kind and g['family']==i for g in data) for i in range(3)] for kind in ['repair','keep']};trace=[]
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);losses=dict(repair=0.,keep=0.)
        for g in data:
            loc=location(q,g)
            if g['kind']=='repair':
                per=P.action_loss(loc,g['target']).mean(-1);loss=.5*per.mean()+.5*per.topk(max(1,math.ceil(.2*len(per)))).values.mean()
            else:loss=(loc-g['target']).square().mean()
            weight=1/(3*counts[g['kind']][g['family']]);(.5*weight*loss).backward();losses[g['kind']]+=weight*float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(step=k,repair_balanced_tail_loss=losses['repair'],keep_balanced_MSE=losses['keep'],gradient_norm=norm))
        assert fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    n=len(trace);assert all(int(ao.state[a]['step'])==640+n for a in q.drive_adapter.parameters()) and all(int(ao.state[a]['step'])==n for a in q.steer_adapter.parameters())
    return dict(Actor_Adam=n,Value_Adam=0,drive_step_before=640,drive_step_after=640+n,new_steer_step_before=0,new_steer_step_after=n,initial=initial,final=measure(q,data),trace=trace,trainable_parameters=11906,teacher_at_deployment=False)
