"""A learned causal drive readout; fixed original steering and GRU backbone.

Training uses verified successful trajectories only. No side/time branch,
teacher, waypoint or map is present in the deployed Actor.
"""
from pathlib import Path
import gzip,io,sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t493_wide8_night'))
import t493_runtime as G
P,torch,np=G.P,G.torch,G.np
digest,sha=G.digest,G.sha
PLAIN_FROZEN=P.frozen

class Actor(P.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed'])
            self.drive_adapter=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).double()
            torch.nn.init.zeros_(self.drive_adapter[-1].weight);torch.nn.init.zeros_(self.drive_adapter[-1].bias)
    def location(self,h,n):
        logits=self.head(h).clone()
        logits[...,0]=logits[...,0]+.3*torch.tanh(self.drive_adapter(torch.cat([h,n],-1))[...,0])
        return 1.25*torch.tanh(logits)
    def sequence(self,x):
        single=x.ndim==2
        if single:x=x[:,None]
        n=self.normalized(x);h,_=self.core(self.encoder(n));loc=self.location(h,n)
        return loc[:,0] if single else loc
    def step(self,x,hidden=None):
        if hidden is None:hidden=x.new_zeros((1,1,64),dtype=torch.float64)
        n=self.normalized(x);h,new=self.core(self.encoder(n)[None],hidden);loc=self.location(h[0],n)
        z=np.zeros((1,2)) if not self.sample else self.rng.normal(size=(1,2))
        raw=loc.detach()+loc.new_tensor(z)*loc.new_tensor(self.p['raw_sigma'])
        atom=torch.where(raw>=P.L.EDGE,1,torch.where(raw<=-P.L.EDGE,-1,0)).to(torch.int8)
        action=raw.clamp(-1,1).float();assert torch.equal(atom.abs()==1,action.abs()==1)
        lp=P.L.density(loc,raw,atom,self.p['raw_sigma']) if self.sample else torch.zeros(1,dtype=torch.float64)
        self.tape.append(dict(innovations=z[0].copy(),raw_actions=raw[0].cpu().numpy().copy(),atom=atom[0].cpu().numpy().copy(),locations=loc[0].detach().cpu().numpy().copy(),logp=lp[0].detach().cpu().numpy().copy(),stochastic=np.asarray(self.sample,np.uint8)))
        self.previous_raw=raw;self.previous_location=loc.detach();return action,loc,new,loc
    def phash(self):return digest(dict(architecture='T496_single120_64carry_5953learned_drive_readout',state=self.state_dict(),sigma=self.p['raw_sigma']))

def make(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=P.checkpoint(p['initial']['path'])['joint_state'];q=Actor(p)
    missing,unexpected=q.load_state_dict(old['actor'],strict=False)
    assert sorted(missing)==sorted('drive_adapter.'+k for k in q.drive_adapter.state_dict()) and not unexpected
    q.requires_grad_(False);q.drive_adapter.requires_grad_(True)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==5953
    v=P.L.Value(q,p['value_seed']);v.load_state_dict(old['value']);v.requires_grad_(False)
    ao=torch.optim.Adam(q.drive_adapter.parameters(),lr=.001,foreach=False,fused=False)
    vo=torch.optim.Adam(v.parameters(),lr=p['value_lr'],foreach=False,fused=False);vo.load_state_dict(old['value_opt'])
    return q,v,ao,vo

def frozen(p,path):
    saved=P.checkpoint(path);state=saved['joint_state']['actor']
    if not any(k.startswith('drive_adapter.') for k in state):return PLAIN_FROZEN(p,path)
    q=Actor(p).eval().requires_grad_(False);q.load_state_dict(state);assert q.phash()==saved['policy_hash'];return q

def export(obj,path,limit):
    state=P.state(*obj);b=io.BytesIO();torch.save(dict(joint_state=state,policy_hash=obj[0].phash(),joint_hash=digest(state),architecture='T496_single120_64carry_5953learned_drive_readout',single_actor=True,teacher_at_deployment=False),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=limit;Path(path).write_bytes(blob)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(state),bytes=len(blob))

def groups(p,rows):
    q=PLAIN_FROZEN(p,p['initial']['path']);result=[]
    with torch.no_grad():
        for r in rows:
            assert r['success'] and not r['collision'] and sha(r['path'])==r['sha256']
            a=G.E.N.S.arrays(r['path']);x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];base=q.head(h)
            # All verified teachers use this very same frozen steering function.
            assert torch.equal((1.25*torch.tanh(base[:,1])).clamp(-1,1).float(),torch.tensor(a['actions'][:,1]))
            result.append(dict(id=r['job_id'],side=0 if '_LEFT_' in r['job_id'] else 1,features=torch.cat([h,n],-1),logit=base[:,0],target=torch.tensor(a['actions'][:,0],dtype=torch.float64),early=torch.tensor(np.r_[0.,np.cumsum(a['native_counts']*.002)[:-1]]<20),source_sha256=r['sha256']))
    assert len(result)==54 and sum(g['side']==0 for g in result)==27
    return result

def backbone(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('drive_adapter.')})

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=backbone(q);vhash=digest(v.state_dict());vh=digest(vo.state_dict());trace=[]
    pools=[[g for g in data if g['side']==s] for s in [0,1]]
    for k in range(p['bc_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=0.
        for pool in pools:
            for j in range(6):
                g=pool[(6*k+j)%27];offset=.3*torch.tanh(q.drive_adapter(g['features'])[:,0]);location=1.25*torch.tanh(g['logit']+offset)
                err=P.action_loss(location[:,None],g['target'][:,None])[:,0]
                loss=(.5*err.mean()+.5*err[g['early']].mean())/12
                loss.backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(q.drive_adapter.parameters(),1.,error_if_nonfinite=True))
        event('BEFORE',k);ao.step();event('AFTER',k);trace.append(dict(step=k,loss=total,norm=norm))
        assert backbone(q)==fixed and digest(v.state_dict())==vhash and digest(vo.state_dict())==vh
    return dict(Actor_Adam=len(trace),Value_Adam=0,trace=trace,trainable_parameters=5953,frozen_backbone_sha256=fixed,teacher_at_deployment=False)
