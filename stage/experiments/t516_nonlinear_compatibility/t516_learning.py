"""Frozen parent, unbounded logit correction, blocked-label offline test."""
from pathlib import Path
import gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t515_decoder_recovery_transfer'))
import t515_learning as D
A,G,P,torch,np=D.A,D.G,D.P,D.torch,D.np
read,sha,digest=G.read,G.sha,G.digest
class Actor(A.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['residual_seed'])
            self.recovery_net=torch.nn.Sequential(torch.nn.Linear(184,64),torch.nn.Tanh(),torch.nn.Linear(64,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double()
            torch.nn.init.zeros_(self.recovery_net[-1].weight);torch.nn.init.zeros_(self.recovery_net[-1].bias)
    def location(self,h,n):
        logits=self.head(h).clone();f=torch.cat([h,n],-1)
        logits[...,0]=logits[...,0]+.3*torch.tanh(self.drive_adapter(f)[...,0])
        logits[...,1]=logits[...,1]+.3*torch.tanh(self.steer_adapter(f)[...,0])
        return 1.25*torch.tanh(logits+self.recovery_net(f))
    def phash(self):return digest(dict(architecture='T516_single120_64carry_frozen_parent_nonlinear_logit_correction',state=self.state_dict(),sigma=self.p['raw_sigma']))
def variables(q):return list(q.recovery_net.parameters())
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('recovery_net.')})
def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    saved=P.checkpoint(p['initial']['path']);old=saved['joint_state'];q=Actor(p)
    expanded=any(k.startswith('recovery_net.') for k in old['actor'])
    missing,unexpected=q.load_state_dict(old['actor'],strict=False);assert not unexpected
    assert not missing if expanded else sorted(missing)==sorted('recovery_net.'+k for k in q.recovery_net.state_dict())
    q.requires_grad_(False);q.recovery_net.requires_grad_(True)
    assert sum(z.numel() for z in variables(q))==13986
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(variables(q),lr=p['residual_lr'],foreach=False,fused=False)
    if expanded:ao.load_state_dict(old['actor_opt'])
    else:assert not ao.state
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    if p.get('initial_joint_locked'):assert digest(P.state(q,v,ao,vo))==p['initial']['joint_hash']
    return q,v,ao,vo
def frozen(p,path):
    saved=P.checkpoint(path)
    if not any(k.startswith('recovery_net.') for k in saved['joint_state']['actor']):return A.frozen(p,path)
    q=Actor(p).eval().requires_grad_(False);q.load_state_dict(saved['joint_state']['actor']);assert q.phash()==saved['policy_hash'];return q
def export(obj,path,limit):
    s=P.state(*obj);b=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),architecture='T516_single120_64carry_frozen_parent_nonlinear_logit_correction',single_actor=True,teacher_at_deployment=False),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=limit;Path(path).write_bytes(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(s),bytes=len(blob))

def data(p):
    src=read(p['dataset_protocol']);assert sha(p['dataset_protocol'])==p['dataset_protocol_sha256']
    assert src['parent_model']==p['parent_model'];rows=D.groups(src)
    parent=A.frozen(p,p['parent_model']['path'])
    with torch.no_grad():
        for g in rows:
            f=g['features'];res=torch.stack([parent.drive_adapter(f)[:,0],parent.steer_adapter(f)[:,0]],-1)
            g['base_logits']=parent.head(f[:,:64])+.3*torch.tanh(res)
            if g['kind']=='repair':
                index=torch.arange(len(f));g['train_mask']=(index//5)%2==0;g['held_mask']=~g['train_mask']
                assert int(g['train_mask'].sum())==30 and int(g['held_mask'].sum())==29
    assert len(rows)==21;return rows
def location(q,g):return 1.25*torch.tanh(g['base_logits']+q.recovery_net(g['features']))
def loss(q,g,p):
    loc=location(q,g);target=g['target'];dt=g['dt'];sigma=loc.new_tensor(p['raw_sigma'])
    if g['kind']=='repair':
        mask=g['train_mask'];loc,target,dt=loc[mask],target[mask],dt[mask]
        err=(.5*P.action_loss(loc,target)/sigma.square()).mean(-1)
    else:err=(.5*((loc-target)/sigma).square()).mean(-1)
    return .5*(err*dt).sum()/dt.sum()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()

def metrics(q,rows,p):
    keep=[];prefix=None;repair=None
    with torch.no_grad():
        for g in rows:
            loc=location(q,g);diff=loc-g['target']
            if g['kind']=='repair':
                diff=loc.clamp(-1,1).float().double()-g['target'];repair={}
                for which in ['train','held']:
                    d=diff[g[which+'_mask']]
                    repair[which]=dict(count=len(d),executed_RMSE_per_axis=d.square().mean(0).sqrt().tolist(),max_abs_per_axis=d.abs().max(0).values.tolist())
            else:
                item=dict(id=g['id'],RMSE_per_axis=diff.square().mean(0).sqrt().tolist(),max_abs=float(diff.abs().max()))
                if g['kind']=='keep':keep.append(item)
                else:prefix=item
    assert len(keep)==19 and prefix is not None and repair is not None
    passed=all(v<=p['teacher_RMSE_limit'] for which in ['train','held'] for v in repair[which]['executed_RMSE_per_axis'])
    passed=passed and all(v<=p['keep_RMSE_limit'] for item in keep+[prefix] for v in item['RMSE_per_axis'])
    passed=passed and all(item['max_abs']<=p['keep_max_abs_limit'] for item in keep+[prefix])
    return dict(repair=repair,keep19=keep,prefix=prefix,offline_joint_gate=bool(passed),not_navigation_or_fresh_episode_holdout=True)

def fit(obj,rows,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict())
    initial=metrics(q,rows,p);trace=[]
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=dict(repair=0.,keep=0.,prefix=0.)
        for g in rows:
            if g['kind']=='repair':weight=.5
            elif g['kind']=='prefix':weight=.1
            else:weight=.4/(2*sum(h['kind']=='keep' and h['family']==g['family'] for h in rows))
            z=loss(q,g,p);(weight*z).backward();total[g['kind']]+=weight*float(z.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(step=k,**total,gradient_norm=norm))
        assert fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    return dict(Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=metrics(q,rows,p),trace=trace,trainable_parameters=13986,frozen_parent_hash=fixed)
