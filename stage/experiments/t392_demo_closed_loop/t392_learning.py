"""Bounded demo-assisted recurrent score-function learning, Stage only.

Forced physical prefixes are training initial-state curricula. Their actions
never enter the likelihood ratio; current-parameter recurrent history does.
"""
from pathlib import Path
import copy,gzip,importlib.util,io,json,math,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t392_demo_closed_loop';DOC=ROOT/'stage/docs/review_20261106'
spec=importlib.util.spec_from_file_location('stage392_existing_native_contract',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
np,torch,read,save,sha,digest,equal=N.np,N.torch,N.read,N.save,N.sha,N.digest,N.equal
PARENT=ROOT/'stage/runs/t391_actual_student_recovery/student/recovery_transfer/candidate.pt.gz'
REFERENCE=ROOT/'stage/runs/t387_full_horizon_night/phase_warp/mixed.npz'
RECOVERY=ROOT/'stage/runs/t391_actual_student_recovery/teacher/mixed.npz'
EDGE=(1.+float(np.nextafter(np.float32(1),np.float32(0))))/2

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes'])
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    return N.science(p)

def logp(loc,raw,atom,p):
    sd=torch.tensor(p['raw_sigma'],dtype=torch.float64)
    raw=raw.detach()
    interior=-.5*((raw-loc)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    return torch.where(atom<0,torch.special.log_ndtr((-EDGE-loc)/sd),
        torch.where(atom>0,torch.special.log_ndtr((loc-EDGE)/sd),interior)).sum(-1)

class Actor(N.S.NightActor):
    def __init__(self,base,p):
        super().__init__(base,0.);self.contract=p;self.sample=False;self.rng=None
        self.load_learned(N.S.read_model(PARENT));self.reset_noise()
    def phash(self):return digest(dict(weights=self.state_dict(),memory=True,distribution='enhanced_float32_clipped_iidGaussian',sigma=self.contract['raw_sigma'],edge=EDGE))
    def step(self,x,hidden=None):
        stochastic=self.sample;self.sample=False
        try:_,loc,new,_=super().step(x,hidden)
        finally:self.sample=stochastic
        z=self.rng.normal(size=(1,2)) if stochastic else np.zeros((1,2))
        raw=loc.detach()+torch.tensor(z,dtype=torch.float64)*torch.tensor(self.contract['raw_sigma'],dtype=torch.float64)
        atom=torch.where(raw>=EDGE,1,torch.where(raw<=-EDGE,-1,0)).to(torch.int8)
        action=raw.clamp(-1,1).float();assert torch.equal(atom.abs()==1,action.abs()==1)
        self.tape[-1].update(innovations=z[0].copy(),raw_actions=raw[0].numpy().copy(),atom=atom[0].numpy().copy(),
            logp=(logp(loc,raw,atom,self.contract) if stochastic else torch.zeros(1))[0].detach().numpy().copy(),
            stochastic=np.asarray(stochastic,np.uint8))
        self.previous_raw=raw;self.previous_location=loc.detach()
        return action,loc,new,loc

def learner(base,p):
    q=Actor(base,p).eval();params=N.trainable(q);assert sum(v.numel() for v in params)==6864
    return q,torch.optim.Adam(params,lr=p['lr'])

def state(q,opt):return N.learned_state(q,opt)
def load(q,opt,s):q.load_learned(s['learned']);opt.load_state_dict(copy.deepcopy(s['optimizer']))
def statehash(q,opt):return {k:digest(v) for k,v in state(q,opt).items()}
def export(q,opt,path):
    b=io.BytesIO();torch.save(dict(state=state(q,opt),hashes=statehash(q,opt),policy_hash=q.phash(),memory_on=True),b)
    data=gzip.compress(b.getvalue(),mtime=0);assert len(data)<=224*1024;Path(path).write_bytes(data)
    return dict(path=str(path),sha256=sha(path),hashes=statehash(q,opt),policy_hash=q.phash())

def datasets(q):
    out=[]
    for f,first in [(REFERENCE,0),(RECOVERY,12)]:
        a=N.S.arrays(f);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
        x=torch.tensor(a['states']);out.append(dict(x=x,b=q.base_actions(x),y=torch.tensor(a['actions'],dtype=torch.float64),first=first))
    return out

def total(a,first):
    assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
    ret=0.
    for r,n in zip(a['rewards'][first:][::-1],a['native_counts'][first:][::-1]):ret=float(r)+N.K.R.GAMMA**int(n)*ret
    return ret

def audit_training(q,a,first,seed,p):
    """Independently replay the Actor and raw innovation stream, zero physics."""
    q.reset_noise();q.rng=np.random.default_rng(np.random.SeedSequence(seed));h=None
    with torch.no_grad():
        for k,row in enumerate(a['states']):
            q.sample=k>=first
            hb=np.zeros(16) if h is None else h[0,0].numpy().copy()
            command,loc,h,_=q.step(torch.tensor(row[None]),h)
            assert equal(hb,a['hidden_before'][k])
            for name,v in q.tape[-1].items():assert equal(np.asarray(v),a[name][k]),(k,name)
            if k>=first:assert equal(command[0].numpy(),a['actions'][k])
        assert equal(h[0,0].numpy(),a['last_hidden_after'])
    q.sample=False
    assert np.array_equal(a['atom'].astype(bool),np.abs(a['raw_actions'])>=EDGE)
    goal=a['task_goal_xy'];prior=N.S.arrays(REFERENCE)
    assert equal(a['actions'][:first],prior['actions'][:first])
    for k,n in enumerate(a['native_counts']):
        flags=a['end_flags'][k];d0=float(np.linalg.norm(a['poses'][k,:2]-goal));d1=float(np.linalg.norm(a['next_poses'][k,:2]-goal))
        z=N.K.R.GAMMA**int(n);rw=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not flags.any())
        if flags.any():rw+=N.K.R.GAMMA**(int(n)-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(rw-a['rewards'][k])<1e-12,(k,rw,a['rewards'][k])

def blocks(q,items,first,p):
    returns=np.asarray([total(a,first) for a in items]);out=[];errors=[]
    for i,a in enumerate(items):
        assert not a['stochastic'][:first].any() and a['stochastic'][first:].all()
        x=torch.tensor(a['states']);b=q.base_actions(x);raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom'])
        old=torch.tensor(a['logp']);adv=float(returns[i]-(returns.sum()-returns[i])/3)
        with torch.no_grad():
            loc=q.sequence(x,b);err=float((loc-torch.tensor(a['locations'])).abs().max())
            lp_err=float((logp(loc,raw,atom,p)[first:]-old[first:]).abs().max())
            assert err<1e-10 and lp_err<1e-9,(err,lp_err)
        out.append((x,b,raw,atom,old,adv));errors.append([err,lp_err])
    return out,returns,errors

def fit(q,opt,items,first,p,refs,guard,event):
    batch,returns,errors=blocks(q,items,first,p);trace=[]
    fixed=digest({k:v for k,v in q.state_dict().items() if k.startswith(('base.','future_head.'))})
    for k in range(p['epochs']):
        guard();opt.zero_grad(set_to_none=True);pg=aux=0.;maxlog=0.
        for x,b,raw,atom,old,adv in batch:
            lr=(logp(q.sequence(x,b),raw,atom,p)-old)[first:];ratio=lr.exp();assert torch.isfinite(ratio).all()
            c=adv/(4*400)
            loss=-torch.minimum(ratio*c,ratio.clamp(1-p['ppo_clip'],1+p['ppo_clip'])*c).sum()
            loss.backward();pg+=float(loss.detach());maxlog=max(maxlog,float(lr.detach().abs().max()))
        for g in refs:
            pred=q.sequence(g['x'],g['b']);loss=.5*p['demo_weight']*(pred[g['first']:]-g['y'][g['first']:]).square().mean()
            loss.backward();aux+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_([v for v in q.parameters() if v.requires_grad],1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert digest({n:v for n,v in q.state_dict().items() if n.startswith(('base.','future_head.'))})==fixed
        trace.append([pg,aux,norm,maxlog])
    with torch.no_grad():
        maxlog=max(float((logp(q.sequence(x,b),raw,atom,p)-old)[first:].abs().max()) for x,b,raw,atom,old,_ in batch)
    return dict(trace=trace,returns=returns.tolist(),density_errors=errors,post_max_abs_logratio=maxlog,
                numeric_accept=bool(np.isfinite(maxlog) and maxlog<=p['max_logratio']))
