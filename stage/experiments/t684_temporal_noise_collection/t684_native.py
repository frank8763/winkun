"""Explicit raw latent noise and actual previous native-hold clock."""
from pathlib import Path
import copy,math,os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t683_temporal_density_contract'))
import t683_runtime as Contract
C,T,M,torch,np=Contract.C,Contract.T,Contract.M,Contract.torch,Contract.np
B=T.N.Base
G,F,E,P,L=B.G,B.F,B.E,B.P,B.L
digest,sha=C.digest,C.sha
BASE_FROZEN=M.L.frozen
SEMANTICS='T684_raw_latent_OU_actual_clock_v1'

def behavior(p,tau):return digest(dict(weight_policy=p['initial']['policy_hash'],sigma=p['raw_sigma'],tau_seconds=float(tau),semantics=SEMANTICS))

def configure(q,tau):
    assert tau in [0.,2.];q.temporal_tau=float(tau)

def frozen(p,path):
    q=BASE_FROZEN(p,path);original_step=q.step;original_reset=q.reset_noise;weight=q.phash()
    assert weight==p['initial']['policy_hash']
    q.temporal_tau=0.;q.temporal_count=None;q.temporal_previous=None
    def reset():
        original_reset();q.temporal_count=None;q.temporal_previous=None
    def step(x,hidden=None):
        assert q.sample and q.rng is not None,'Only registered stochastic collection uses this loader'
        q.sample=False
        try:unused,loc,new,extra=original_step(x,hidden)
        finally:q.sample=True
        q.tape.pop() # Replace deterministic proposal, preserving only the actual draw.
        if q.temporal_previous is None:rho=0.;prev=np.zeros((1,2));dt=0.
        else:
            assert q.temporal_count is not None and 0<q.temporal_count<=100
            dt=float(q.temporal_count)*.002;rho=math.exp(-dt/q.temporal_tau) if q.temporal_tau else 0.;prev=q.temporal_previous
        epsilon=q.rng.normal(size=(1,2));z=rho*prev+math.sqrt(1-rho*rho)*epsilon
        sigma=loc.new_tensor(p['raw_sigma']);raw=loc.detach()+loc.new_tensor(z)*sigma
        cm=loc.detach()+loc.new_tensor(rho*prev)*sigma;cs=sigma*math.sqrt(1-rho*rho)
        atom=torch.where(raw>=M.P.L.EDGE,1,torch.where(raw<=-M.P.L.EDGE,-1,0)).to(torch.int8)
        action=raw.clamp(-1,1).float();assert torch.equal(atom.abs()==1,action.abs()==1)
        latent=torch.distributions.Normal(cm,cs).log_prob(raw).sum(-1)
        clipped=M.P.L.density(cm,raw,atom,cs.tolist())
        q.tape.append(dict(innovations=epsilon[0].copy(),noise_latent=z[0].copy(),raw_actions=raw[0].numpy().copy(),
            locations=loc[0].detach().numpy().copy(),atom=atom[0].numpy().copy(),stochastic=np.asarray(1,np.uint8),
            logp=latent[0].numpy().copy(),latent_logp=latent[0].numpy().copy(),conditional_clip_logp=clipped[0].numpy().copy(),
            temporal_rho=np.asarray(rho),temporal_tau=np.asarray(q.temporal_tau),previous_hold_seconds=np.asarray(dt),
            conditional_sigma=cs.numpy().copy(),logp_is_raw_latent=np.asarray(1,np.uint8)))
        q.temporal_previous=z.copy();q.previous_raw=raw;q.previous_location=loc.detach()
        return action,loc,new,extra
    q.reset_noise=reset;q.step=step;q.phash=lambda:behavior(p,q.temporal_tau)
    return q

class Session(B.Session):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        assert self.q is not None and self.pn==0 and not self.hidden.any()
        configure(self.q,float(job['temporal_tau']))
        assert self.q.temporal_previous is None
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0 and actor_active
        self.q.temporal_count=int(self.pn) if self.rows else None
        before=len(self.rows);value=super().step(forced=None,owner=0,actor_active=True)
        assert len(self.rows)==before+1 and self.pn==int(self.rows[-1][2])
        assert float(self.proposal[-1]['previous_hold_seconds'])==(0. if before==0 else int(self.rows[-2][2])*.002)
        return value

def worker(pipe,index,p,counts):
    B.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:B.C.consistent_reset.setup(F,folder,pc,B.C.CANONICAL_SETUP)
    P.frozen=frozen;G.T.worker(pipe,index,p,counts)

def inspect(p,a,r):
    """Full physics/history/reward audit with temporal density semantics explicit."""
    assert not a['ownership'].any() and a['stochastic'].all() and a['logp_is_raw_latent'].all()
    assert np.array_equal(a['actions'],np.clip(a['raw_actions'],-1,1).astype(np.float32))
    assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1 and r['complete']
    assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
    assert (a['native_counts'][:-1]==100).all() and int(a['native_counts'].sum())==r['physics']
    tau=float(r['request']['job']['temporal_tau']);q=frozen(p,r['request']['model']);configure(q,tau)
    q.rng=np.random.default_rng(np.random.SeedSequence(r['seed']));q.reset_noise();q.sample=True;hidden=None
    for i,x in enumerate(a['states']):
        q.temporal_count=None if i==0 else int(a['native_counts'][i-1])
        hb=np.zeros(64) if hidden is None else hidden[0,0].numpy().copy()
        with torch.no_grad():action,loc,hidden,unused=q.step(torch.tensor(x[None]),hidden)
        assert np.array_equal(hb,a['hidden_before'][i]) and np.array_equal(action[0].numpy(),a['actions'][i])
        for k,v in q.tape[-1].items():assert np.array_equal(v,a[k][i]),(i,k)
        assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
        n=int(a['native_counts'][i]);gamma=E.N.K.R.GAMMA;z=gamma**n;f=a['end_flags'][i]
        d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy'])
        reward=-.000002*(1-z)/(1-gamma)+.03*d0-z*.03*d1*(not f.any())
        if f.any():reward+=gamma**(n-1)*(float(f[1])-float(f[0])-.5*float(f[2]))
        assert abs(reward-a['rewards'][i])<1e-12
    assert np.array_equal(hidden[0,0].numpy(),a['last_hidden_after']) and q.phash()==r['policy_hash']
    assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
    with torch.no_grad():mu=q.sequence(torch.tensor(a['states']))
    terms=Contract.latent_terms(mu,torch.tensor(a['raw_actions']),a['native_counts'],p['raw_sigma'],tau).numpy()
    assert np.max(np.abs(terms-a['logp']))<1e-9 and np.array_equal(a['logp'],a['latent_logp'])
