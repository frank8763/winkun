"""Bounded paired CAPG/PPO recovery pilot. Published methods, no novelty claim."""
from pathlib import Path
import sys,copy,gzip,io,json,os,time,traceback,shutil,math,contextlib,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t338_capg_recovery';DOC=ROOT/'stage/docs/review_20261104'
sys.path.insert(0,str(ROOT/'stage/experiments/t336_reverse_online'))
import t336_core as H
K,L,N,np,torch=H.K,H.L,H.N,H.np,H.torch
read,save,digest,arrays,packed=H.read,H.save,H.digest,H.arrays,H.packed
# round-to-nearest float32 makes the entire tail beyond this midpoint execute
# exactly 1. Interior raw draws remain observed latent variables: their density
# is NOT misrepresented as the mass of a quantized interior action.
EDGE=(1.+float(np.nextafter(np.float32(1),np.float32(0))))/2

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def clipped_std(mu,sigma):
    low=(-1-mu)/sigma;high=(1-mu)/sigma
    cdf=lambda x:.5*torch.erfc(-x/math.sqrt(2))
    pdf=lambda x:torch.exp(-.5*x.square())/math.sqrt(2*math.pi)
    mass=cdf(high)-cdf(low);fl,fh=pdf(low),pdf(high)
    e=-cdf(low)+mu*mass+sigma*(fl-fh)+(1-cdf(high))
    e2=cdf(low)+(mu.square()+sigma.square())*mass+2*mu*sigma*(fl-fh)+sigma.square()*(low*fl-high*fh)+(1-cdf(high))
    return (e2-e.square()).clamp_min(0).sqrt()

class Actor(torch.nn.Module):
    def __init__(self,p,arm):
        super().__init__();assert arm in p['arms'];self.arm=arm;self.p=p;self.memory_on=False
        self.fixed=L.model(read(p['learning_protocol']))
        L.load(self.fixed,L.read_patch(p['initial_patch'])['residual'])
        assert digest(self.fixed.state_dict())==p['initial_policy_hash']
        self.fixed.requires_grad_(False).eval()
        self.register_buffer('distribution_id',torch.tensor(0 if arm=='tanh' else 1))
        self.register_buffer('mean',torch.tensor(p['input_mean']))
        self.register_buffer('scale',torch.tensor(p['input_scale']))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['seed'])
            self.correction=torch.nn.Sequential(torch.nn.Linear(120,64),torch.nn.Tanh(),torch.nn.Linear(64,2))
            torch.nn.init.zeros_(self.correction[-1].weight);torch.nn.init.zeros_(self.correction[-1].bias)
        self.logscale=torch.nn.Parameter(torch.zeros(2))
    def cache(self,x):
        with torch.no_grad():
            parent=torch.cat([self.fixed.step(row[None])[0] for row in x])
            # Both arms exactly share T336's initialized deterministic control.
            b=torch.tanh(torch.atanh(parent.clamp(-.999999,.999999))).double()
            if self.arm=='tanh':return (b,torch.full_like(b,self.p['tanh_sigma']))
            target=torch.tensor(self.p['executed_std'],dtype=torch.float64).expand_as(b)
            lo=torch.full_like(b,1e-5);hi=torch.ones_like(b)
            for _ in range(48):
                mid=(lo+hi)/2;less=clipped_std(b,mid)<target
                lo=torch.where(less,mid,lo);hi=torch.where(less,hi,mid)
            sigma=(lo+hi)/2
            assert float((clipped_std(b,sigma)-target).abs().max())<1e-8
            return b,sigma
    def stats(self,x,cache=None):
        assert int(self.distribution_id)==(0 if self.arm=='tanh' else 1)
        b,base_sigma=self.cache(x) if cache is None else cache
        norm=((x-self.mean)/self.scale).clamp(-10,10)
        # Same actual batch1 numerical path in worker, learner, and audit.
        d=torch.cat([self.correction(row[None]) for row in norm]).double()
        if self.arm=='tanh':loc=torch.atanh(b.clamp(-.999999,.999999))+d
        else:
            margin=1+base_sigma;loc=margin*torch.tanh(torch.atanh(b/margin)+d)
        sigma=base_sigma*self.logscale.double().clamp(math.log(.5),math.log(2.)).exp()
        return loc,sigma
    def step(self,x,hidden=None):
        loc,sigma=self.stats(x);mode=(torch.tanh(loc) if self.arm=='tanh' else loc.clamp(-1,1)).float()
        return mode,loc,None,loc
    def logp(self,loc,sigma,raw,atom):
        normal=-.5*((raw-loc)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)
        if self.arm=='tanh':
            jac=2*(math.log(2)-raw-torch.nn.functional.softplus(-2*raw))
            return (normal-jac).sum(-1)
        left=torch.special.log_ndtr((-EDGE-loc)/sigma)
        right=torch.special.log_ndtr((loc-EDGE)/sigma)
        return torch.where(atom<0,left,torch.where(atom>0,right,normal)).sum(-1)
    def draw(self,x,z,deterministic=False):
        loc,sigma=self.stats(x);raw=loc if deterministic else loc+sigma*z.double()
        executed=(torch.tanh(raw) if self.arm=='tanh' else raw.clamp(-1,1)).float()
        atom=torch.zeros_like(raw,dtype=torch.int8) if self.arm=='tanh' else torch.where(raw>=EDGE,1,torch.where(raw<=-EDGE,-1,0)).to(torch.int8)
        if self.arm=='capg':assert torch.equal(atom.abs()==1,executed.abs()==1)
        lp=torch.zeros(len(x),dtype=torch.float64) if deterministic else self.logp(loc,sigma,raw,atom)
        return dict(actions=executed,raw_actions=raw,atom=atom,locations=loc,sigmas=sigma,logp=lp)
    def trainable(self):return [*self.correction.parameters(),self.logscale]

class Learner:
    def __init__(self,p,arm):
        self.actor=Actor(p,arm);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0
    def state(self):return dict(actor={k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('fixed.')},optimizer=copy.deepcopy(self.opt.state_dict()))
    def load(self,s):
        self.actor.load_state_dict(dict(self.actor.state_dict(),**s['actor']),strict=True);self.opt.load_state_dict(s['optimizer'])
    def hashes(self):return {k:digest(v) for k,v in self.state().items()}

def returns(a):
    assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
    g=0.;out=np.empty(len(a['rewards']),np.float64)
    for k in range(len(out)-1,-1,-1):
        g=float(a['rewards'][k])+K.R.GAMMA**int(a['native_counts'][k])*g;out[k]=g
    return out

def optimize(p,learner,items,guard,event,steps=None):
    assert len(items)==4;data=[i['arrays'] for i in items];ret=[returns(a) for a in data]
    # Independent trajectory peers, same physical time after the fixed prefix.
    # Every nonfinal decision holds100native steps. Terminated peers have G=0.
    horizon=max(map(len,ret));rms=max(.05,float(np.sqrt(np.mean(np.concatenate(ret)**2))))
    table=np.zeros((4,horizon))
    for j,r in enumerate(ret):table[j,:len(r)]=r
    blocks=[]
    for j,(a,r) in enumerate(zip(data,ret)):
        assert (a['native_counts'][:-1]==100).all()
        x=torch.tensor(a['states']);cache=learner.actor.cache(x)
        raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
        with torch.no_grad():
            loc,sd=learner.actor.stats(x,cache);lp=learner.actor.logp(loc,sd,raw,atom)
            assert torch.equal(lp,old) and np.array_equal(loc.numpy(),a['locations']) and np.array_equal(sd.numpy(),a['sigmas'])
        base=(table.sum(0)-table[j])/3
        discount=K.R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        coeff=torch.tensor(discount*(r-base[:len(r)])/rms/(4*p['suffix_max_decisions']))
        blocks.append((x,cache,raw,atom,old,coeff))
    trace=[]
    for step in range(p['epochs'] if steps is None else steps):
        guard();learner.opt.zero_grad(set_to_none=True);lossvalue=0.;maxlog=0.
        for x,cache,raw,atom,old,c in blocks:
            loc,sd=learner.actor.stats(x,cache);logratio=learner.actor.logp(loc,sd,raw,atom)-old
            assert torch.isfinite(logratio).all();ratio=logratio.exp();assert torch.isfinite(ratio).all()
            objective=torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c)
            loss=-objective.sum();loss.backward();lossvalue+=float(loss.detach());maxlog=max(maxlog,float(logratio.detach().abs().max()))
        norm=float(torch.nn.utils.clip_grad_norm_(learner.actor.trainable(),1.,error_if_nonfinite=True))
        event(dict(stage='BEFORE_ADAM',step=step,hashes=learner.hashes()))
        learner.opt.step();learner.calls+=1
        assert all(v.grad is None for v in learner.actor.fixed.parameters())
        row=dict(step=step,loss=lossvalue,gradient_norm=norm,max_abs_logratio=maxlog,hashes=learner.hashes())
        trace.append(row);event(dict(stage='AFTER_ADAM',**row))
    return trace

def export(p,learner,path):
    buf=io.BytesIO();torch.save(dict(state=learner.state(),hashes=learner.hashes(),arm=learner.actor.arm),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];Path(path).write_bytes(blob)
    return dict(path=str(path),bytes=len(blob),sha256=K.C.sha(path),hashes=learner.hashes())
