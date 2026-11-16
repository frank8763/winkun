"""Frozen complete T550 plus a separately learned causal recurrent residual.

Supervised recovery qualification only. No new RL, critic or runtime teacher.
The two auxiliary GRU memories are explicit in every step's recorded tape.
"""
from pathlib import Path
import sys, copy, gzip, io, math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t596_recurrent_joint_online'))
import t596_learning as Previous
Parent=Previous.Parent
P,G,torch,np=Previous.P,Previous.G,Previous.torch,Previous.np
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
ARCH='T598_frozen_T550_independent32GRU_difference_residual'


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Sequential(torch.nn.Linear(120,32),torch.nn.Tanh())
        self.core=torch.nn.GRU(32,32)
        self.head=torch.nn.Linear(32,2)
        self.double()
    def forward(self,n,h=None):
        z,h=self.core(self.encoder(n),h)
        return self.head(z),h


class Actor(Parent.Actor):
    def __init__(self,p,old_n,added_n,sizes):
        super().__init__(p,old_n,added_n,sizes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(59820261116)
            self.branch=Branch()
        self.anchor=copy.deepcopy(self.branch)
        self.requires_grad_(False)
        self.extra_hidden=self.anchor_hidden=None
    def reset_noise(self):
        super().reset_noise()
        self.extra_hidden=self.anchor_hidden=None
    def sequence(self,x):
        base=super().sequence(x);single=x.ndim==2
        n=self.normalized(x[:,None] if single else x)
        z,_=self.branch(n);z0,_=self.anchor(n)
        delta=z-z0
        return base+(delta[:,0] if single else delta)
    def step(self,x,hidden=None):
        # Legacy hidden remains64; both extra32 memories are separately recorded.
        if hidden is None:self.extra_hidden=self.anchor_hidden=None
        eb=x.new_zeros(32) if self.extra_hidden is None else self.extra_hidden[0,0].clone()
        ab=x.new_zeros(32) if self.anchor_hidden is None else self.anchor_hidden[0,0].clone()
        _,base,new,_=super().step(x,hidden)
        n=self.normalized(x)[None]
        z,self.extra_hidden=self.branch(n,self.extra_hidden)
        z0,self.anchor_hidden=self.anchor(n,self.anchor_hidden)
        loc=base+(z[0]-z0[0])
        row=self.tape[-1];innovation=loc.new_tensor(row['innovations'][None])
        raw=loc.detach()+innovation*loc.new_tensor(self.p['raw_sigma'])
        atom=torch.where(raw>=P.L.EDGE,1,torch.where(raw<=-P.L.EDGE,-1,0)).to(torch.int8)
        action=raw.clamp(-1,1).float()
        lp=P.L.density(loc,raw,atom,self.p['raw_sigma']) if self.sample else loc.new_zeros(1)
        row.update(raw_actions=raw[0].detach().numpy().copy(),atom=atom[0].numpy().copy(),
            locations=loc[0].detach().numpy().copy(),logp=lp[0].detach().numpy().copy(),
            extra_hidden_before=eb.detach().numpy().copy(),anchor_hidden_before=ab.detach().numpy().copy())
        self.previous_raw=raw;self.previous_location=loc.detach()
        return action,loc,new,loc
    def phash(self):
        return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))


def initial(p):
    assert sha(p['base_model']['path'])==p['base_model']['sha256']
    base=Parent.frozen(p,p['base_model']['path'])
    q=Actor(p,len(base.repair_centers),len(base.added_centers),[len(z.centers) for z in base.layers])
    missing,extra=q.load_state_dict(base.state_dict(),strict=False)
    assert not extra and set(missing)=={k for k in q.state_dict() if k.startswith(('branch.','anchor.'))}
    assert all(torch.equal(q.state_dict()[k],v) for k,v in base.state_dict().items())
    return q.eval()


def frozen(p,path):
    s=P.checkpoint(path)
    if s.get('architecture')!=ARCH:return Parent.frozen(p,path)
    assert s['base_model']==p['base_model']
    q=initial(p);missing,extra=q.load_state_dict(s['residual_state'],strict=False)
    assert not extra and set(missing)=={k for k in q.state_dict() if not k.startswith(('branch.','anchor.'))}
    assert q.phash()==s['policy_hash'] and fixed(q)==s['fixed_sha']
    return q


def variables(q):return list(q.branch.parameters())
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('branch.')})
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def make(p):
    q=initial(p);q.branch.requires_grad_(True)
    return q,torch.optim.Adam(variables(q),lr=p['fit_lr'],foreach=False,fused=False)


def dataset(p):
    data=Previous.references(p);q=initial(p)
    with torch.no_grad():
        for g in data:
            g['n']=q.normalized(g['x'])
            # Full prior history is retained, including all forced prefix inputs.
            g['base']=Parent.Actor.sequence(q,g['x']).detach()
            g['reference']=q.anchor(g['n'][:,None])[0][:,0].detach()
    return data


def measure(q,data):
    rows=[]
    with torch.no_grad():
        for g in data:
            z=q.branch(g['n'][:,None])[0][:,0]
            loc=g['base']+(z-g['reference'])
            target=g['target'][g['mask']].clamp(-1,1).float().double()
            err=loc[g['mask']].clamp(-1,1).float().double()-target
            rows.append(dict(id=g['id'],kind=g['kind'],MAE=float(err.abs().mean()),RMS=float(err.square().mean().sqrt()),max=float(err.abs().max())))
    return rows


def objective(q,data,packed):
    pred=q.branch(packed)[0];loss=packed.new_zeros(());parts=dict(old=0.,recovery=0.)
    for i,g in enumerate(data):
        loc=(g['base']+(pred[:len(g['n']),i]-g['reference']))[g['mask']]
        t=g['target'][g['mask']]
        per=(loc-t).square().mean(-1) if g['kind']=='old' else P.action_loss(loc,t).mean(-1)
        part=(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())*.5/(39 if g['kind']=='old' else 3)
        loss=loss+part;parts[g['kind']]+=float(part.detach())
    return loss,parts


def fit(obj,data,p,guard,event,steps=None):
    q,opt=obj;body=fixed(q);before=measure(q,data);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);trace=[]
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        trace.append(dict(step=k,losses=parts,gradient_norm=norm))
    assert fixed(q)==body
    return dict(steps=len(trace),trace=trace,before=before,after=measure(q,data),fixed_sha=body,
        trainable_parameters=sum(v.numel() for v in variables(q)),new_RL=0,new_Critic=0,
        full_current_residual_history_BPTT=True,not_single_module_causal_ablation=True)


def export(obj,p,path):
    q,opt=obj;b=io.BytesIO();s=state(obj)
    torch.save(dict(architecture=ARCH,base_model=p['base_model'],residual_state={k:v for k,v in s['actor'].items() if k.startswith(('branch.','anchor.'))},actor_opt=s['actor_opt'],joint_hash=digest(s),fixed_sha=fixed(q),policy_hash=q.phash()),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with Path(path).open('xb') as f:f.write(blob)
    return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
