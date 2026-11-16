"""One learned Actor: exact T550 body plus zero-initialized smooth RL residual.

The original neural basis readouts remain in the same forward graph. No task
selector, external controller, teacher, or new observation exists at runtime.
"""
from pathlib import Path
import copy
import gzip
import io
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'))
import t550_actor as Parent
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'))
import t519_learning as Credit
P,G,torch,np=Parent.P,Parent.G,Parent.torch,Parent.np
read,save,sha,digest=Parent.read,Parent.save,Parent.sha,Parent.digest
ORIGINAL_FROZEN=Parent.frozen
ARCH='T581_SINGLE_ACTOR_EXACT_T550_PLUS_BOUNDED_NEURAL_RESIDUAL'


class Actor(Parent.Actor):
    def __init__(self,p,parent):
        super().__init__(p,len(parent.repair_centers),len(parent.added_centers),
                         [len(z.centers) for z in parent.layers])
        self.load_state_dict(parent.state_dict())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['residual_seed'])
            self.continual=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double()
            torch.nn.init.zeros_(self.continual[-1].weight)
            torch.nn.init.zeros_(self.continual[-1].bias)
        self.requires_grad_(False)

    def location(self,h,n):
        return super().location(h,n)+self.p['residual_scale']*torch.tanh(self.continual(torch.cat([h,n],-1)))

    def phash(self):
        return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma'],scale=self.p['residual_scale']))


def fresh(p):
    assert sha(p['base_model']['path'])==p['base_model']['sha256']
    base=ORIGINAL_FROZEN(p,p['base_model']['path'])
    assert base.phash()==p['base_model']['policy_hash']
    return Actor(p,base)


def frozen(p,path):
    s=P.checkpoint(path)
    if s.get('architecture')!=ARCH:return ORIGINAL_FROZEN(p,path)
    assert s['base_model']==p['base_model']
    q=fresh(p);q.continual.load_state_dict(s['continual'])
    assert q.phash()==s['policy_hash']
    return q.eval()


def make(p):
    q=fresh(p);q.continual.requires_grad_(True)
    assert sum(z.numel() for z in q.parameters() if z.requires_grad)==5986
    return q,torch.optim.Adam(q.continual.parameters(),lr=p['rl_lr'],foreach=False,fused=False)


def state(obj):
    return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))


def load(obj,s):
    obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])


def fixed(q):
    return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.')})


def feature(q,x):
    with torch.no_grad():
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        return torch.cat([h,n],-1),Parent.Actor.location(q,h,n)


def loc(q,f,b):
    return b+q.p['residual_scale']*torch.tanh(q.continual(f))


def references(p):
    q=fresh(p);data=[]
    with torch.no_grad():
        for row in p['keep_rows']:
            assert sha(row['path'])==row['sha256']
            a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);f,b=feature(q,x)
            assert torch.equal(loc(q,f,b),q.sequence(x))
            data.append(dict(id=row['job_id'],f=f,base=b,target=b.clone()))
    assert len(data)==27
    return data


def fit(obj,arrays,data,p,guard,event,steps=None):
    q,opt=obj;body=fixed(q);items,diag=Credit.peer_batch(obj,arrays,p)
    cache=[]
    for b in items:
        f,base=feature(q,b['x']);cache.append(dict(b,f=f,base=base))
    trace=[]
    for k in range(p['rl_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);pl=keep=0.
        for b in cache:
            z=loc(q,b['f'],b['base'])
            ratio=(P.L.density(z,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
            loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cache)
            loss.backward();pl+=float(loss.detach())
        for g in data:
            loss=.5*((loc(q,g['f'],g['base'])-g['target'])/q.head.weight.new_tensor(p['raw_sigma'])).square().mean()/len(data)
            (p['keep_weight']*loss).backward();keep+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        with torch.no_grad():
            changes=[P.L.density(loc(q,b['f'],b['base']),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cache]
            kl=sum(float((v.exp()-1-v).mean()) for v in changes)/len(changes)
        trace.append(dict(policy_loss=pl,keep_loss=keep,gradient_norm=norm,KL=kl))
        if kl>p['target_kl']:break
    assert fixed(q)==body
    return dict(trace=trace,Actor_Adam=len(trace),Critic_Adam=0,peer_batch=diag,
                frozen_T550_body_hash=body,new_BC=0,full_actor_trainable=False,
                bounded_residual_NOT_closed_loop_retention_guarantee=True)


def export(obj,p,path):
    q,opt=obj;s=state(obj);buf=io.BytesIO()
    torch.save(dict(architecture=ARCH,base_model=p['base_model'],continual=q.continual.state_dict(),
                    actor_opt=s['actor_opt'],policy_hash=q.phash(),joint_hash=digest(s)),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
    return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
