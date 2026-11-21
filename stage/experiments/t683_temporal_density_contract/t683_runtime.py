"""Latent temporal Gaussian semantics only: no native stepping or optimization."""
from pathlib import Path
import fcntl,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t683_temporal_density_contract';DOC=ROOT/'stage/docs/weekend_20261120_23/t683'
import t683_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t679_episode_parameter_exploration'))
import t679_calibration as C
T,M,torch,np=C.T,C.M,C.torch,C.np
read,sha,digest=C.read,C.sha,C.digest

def coefficients(counts,tau):
    counts=np.asarray(counts,dtype=np.float64);assert counts.ndim==1 and np.isfinite(counts).all() and (counts>0).all()
    return np.r_[0.,np.exp(-counts[:-1]*.002/tau)] if tau else np.zeros(len(counts))

def latent_terms(mu,raw,counts,sigma,tau):
    rho=mu.new_tensor(coefficients(counts,tau))[:,None]
    prior_raw=torch.cat([raw.new_zeros((1,2)),raw[:-1]])
    prior_mu=torch.cat([mu.new_zeros((1,2)),mu[:-1]])
    mean=mu+rho*(prior_raw-prior_mu)
    sd=mu.new_tensor(sigma)*torch.sqrt(1-rho.square())
    return torch.distributions.Normal(mean,sd).log_prob(raw).sum(-1)

class Sampler:
    """Explicit physical clock adapter, not yet a native-worker integration."""
    def __init__(self,q,tau,seed):self.q=q;self.tau=tau;self.reset(seed)
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.hidden=None;self.previous=None
        self.q.sample=False;self.q.tape=[]
    def step(self,x,previous_count=None):
        with torch.no_grad():unused,loc,self.hidden,unused2=self.q.step(x,self.hidden)
        if self.previous is None:rho=0.
        else:
            assert previous_count is not None and previous_count>0
            rho=math.exp(-previous_count*.002/self.tau) if self.tau else 0.
        previous=np.zeros((1,2)) if self.previous is None else self.previous
        epsilon=self.rng.normal(size=(1,2));noise=rho*previous+math.sqrt(1-rho*rho)*epsilon
        sigma=loc.new_tensor(self.q.p['raw_sigma']);raw=loc.detach()+loc.new_tensor(noise)*sigma
        conditional=loc.detach()+loc.new_tensor(rho*previous)*sigma
        sd=sigma*math.sqrt(1-rho*rho)
        atom=torch.where(raw>=M.P.L.EDGE,1,torch.where(raw<=-M.P.L.EDGE,-1,0)).to(torch.int8)
        command=raw.clamp(-1,1).float();assert torch.equal(command.abs()==1,atom.abs()==1)
        conditional_clip=M.P.L.density(conditional,raw,atom,sd.tolist())
        latent=torch.distributions.Normal(conditional,sd).log_prob(raw).sum(-1)
        self.previous=noise.copy()
        return dict(mu=loc[0].detach().numpy().copy(),epsilon=epsilon[0],noise=noise[0],raw=raw[0].numpy().copy(),
            command=command[0].numpy().copy(),atom=atom[0].numpy().copy(),rho=np.asarray(rho),
            conditional_sigma=sd.numpy().copy(),conditional_clip_logp=conditional_clip[0].numpy().copy(),latent_logp=latent[0].numpy().copy())

def trajectory(q,states,counts,tau,seed):
    sampler=Sampler(q,tau,seed);rows=[]
    for i,x in enumerate(states):rows.append(sampler.step(torch.tensor(x)[None],None if i==0 else int(counts[i-1])))
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}

def fixtures():
    # Fixed-order quadrature needs only the existing NumPy runtime. The
    # omitted standard Gaussian tails beyond |12| are <4e-33.
    def pdf(x,mean=0.,sd=1.):return math.exp(-.5*((float(x)-mean)/sd)**2)/(sd*math.sqrt(2*math.pi))
    def sf(x,mean=0.,sd=1.):return .5*math.erfc((float(x)-mean)/(sd*math.sqrt(2)))
    def cdf(x,mean=0.,sd=1.):return .5*math.erfc(-(float(x)-mean)/(sd*math.sqrt(2)))
    def quad(fn,left,right,n=256):
        x,w=np.polynomial.legendre.leggauss(n);mid=(left+right)/2;half=(right-left)/2
        return half*sum(float(weight)*fn(mid+half*float(node)) for node,weight in zip(x,w))
    sigma=.05;mu=.95;rho=math.exp(-.2/2);edge=float(M.P.L.EDGE);sd=sigma*math.sqrt(1-rho*rho)
    conditional_mu=mu+rho*(1.01-mu)
    # Integrate in standardized coordinates, avoiding an unnecessarily narrow
    # Gaussian spike in command coordinates.
    mass=quad(pdf,max(-12.,(-edge-conditional_mu)/sd),min(12.,(edge-conditional_mu)/sd))+cdf(-edge,conditional_mu,sd)+sf(edge,conditional_mu,sd)
    lower=(edge-mu)/sigma
    integrand=lambda u:pdf(u)*sf((edge-mu-rho*sigma*u)/sd)
    joint=quad(integrand,lower,12.)
    coarse=quad(integrand,lower,12.,128)
    naive=sf(lower)*sf((edge-conditional_mu)/sd)
    raw=np.array([[.97,1.01],[1.03,.94]]);means=np.full((2,2),mu)
    value=float(latent_terms(torch.tensor(means),torch.tensor(raw),np.array([100,100]),[sigma,sigma],2.).sum())
    covariance=sigma**2*np.array([[1.,rho],[rho,1.]])
    inv=np.linalg.inv(covariance);logdet=float(np.linalg.slogdet(covariance)[1])
    direct=sum(-math.log(2*math.pi)-.5*logdet-.5*float((raw[:,j]-means[:,j])@inv@(raw[:,j]-means[:,j])) for j in range(2))
    corner=quad(lambda u:pdf(u)*sf(-rho*u/math.sqrt(1-rho*rho)),0.,12.)
    corner_exact=.25+math.asin(rho)/(2*math.pi)
    assert abs(mass-1)<1e-10 and abs(value-direct)<1e-10 and abs(joint-naive)>1e-4 and abs(corner-corner_exact)<1e-10 and abs(joint-coarse)<1e-10
    return dict(one_step_clipped_mixture_mass=float(mass),raw_two_time_logdensity_error=abs(value-direct),
        upper_atom_joint_integrated=float(joint),naive_single_raw_atom_product=float(naive),difference=abs(joint-naive),
        exact_zero_threshold_corner=corner_exact,quadrature_corner_error=abs(corner-corner_exact),quadrature128_vs256_error=abs(joint-coarse),
        clipping_EDGE=edge,conditional_atoms_not_marginal_command_history_likelihood=True)

def compute(p,short=False,guard=lambda:None):
    science=read(p['science_protocol']);q=M.L.frozen(science,p['initial']['path']);before=digest(q.state_dict())
    rows=p['history_rows'][:1] if short else p['history_rows'];arrays={};metrics=[];all_fixed=True
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for index,row in enumerate(rows):
        guard();assert sha(row['path'])==row['sha256'];a=T.R.E.N.S.arrays(row['path']);assert not a['ownership'].any()
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        states=a['states'][:32] if short else a['states'];counts=a['native_counts'][:len(states)]
        fixed=bool((np.asarray(a['native_counts'])[:-1]==100).all());all_fixed=all_fixed and fixed
        seed=[68320261121,0,index]
        for tau in [0.,2.]:
            sample=trajectory(q,states,counts,tau,seed)
            reset=trajectory(q,states[:min(8,len(states))],counts[:min(8,len(states))],tau,seed)
            assert all(np.array_equal(v[:len(reset[k])],reset[k]) for k,v in sample.items())
            rho=coefficients(counts,tau);assert np.array_equal(sample['rho'],rho)
            with torch.no_grad():full=q.sequence(torch.tensor(states)).numpy()
            assert np.max(np.abs(full-sample['mu']))<1e-10
            terms=latent_terms(torch.tensor(sample['mu']),torch.tensor(sample['raw']),counts,science['raw_sigma'],tau).numpy()
            assert np.max(np.abs(terms-sample['latent_logp']))<1e-10
            if not tau:
                reference=M.L.frozen(science,p['initial']['path']);reference.sample=True;reference.rng=np.random.default_rng(seed);reference.tape=[];hidden=None
                for x in states:
                    with torch.no_grad():command,unused,hidden,unused2=reference.step(torch.tensor(x)[None],hidden)
                assert np.array_equal(np.stack([v['raw_actions'] for v in reference.tape]),sample['raw'])
                assert np.max(np.abs(np.stack([v['logp'] for v in reference.tape])-sample['conditional_clip_logp']))<1e-10
            for key,v in sample.items():arrays[f'h{index}_tau{int(tau)}_{key}']=v
            metrics.append(dict(id=row['job_id'],source_sha256=row['sha256'],tau=tau,decisions=len(states),fixed_nonterminal_clock=fixed,
                nonterminal_native_counts=sorted(set(map(int,a['native_counts'][:-1]))),clip_atoms=int(np.count_nonzero(sample['atom'])),
                latent_density_sum=float(terms.sum()),conditional_clip_logp_sum=float(sample['conditional_clip_logp'].sum()),
                sums_have_different_meanings_not_interchangeable=True,external_history_not_environment_rollout=True))
    guard();row=rows[0];a=T.R.E.N.S.arrays(row['path']);states=a['states'][:32] if short else a['states'];counts=a['native_counts'][:len(states)]
    sub=dict(a,states=states);g=M.features(q,sub,row['job_id'],row['sha256']);theta,layout=C.parameter_layout(q)
    raw=torch.tensor(arrays['h0_tau2_raw']);theta=theta.detach().requires_grad_(True)
    def loss(th):return latent_terms(C.output(q,g,th,layout),raw,counts,science['raw_sigma'],2.).mean()
    score=torch.autograd.grad(loss(theta),theta)[0].detach();numeric=[]
    for i in range(3):
        guard();direction=theta.new_tensor(C.direction([68320261121,3,i]));direction=direction/direction.norm();eps=p['numeric_epsilon']
        actual=float((loss(theta+eps*direction)-loss(theta-eps*direction))/(2*eps));expected=float(score@direction)
        error=abs(actual-expected);passed=error<=p['numeric_atol']+p['numeric_rtol']*abs(expected)
        numeric.append(dict(index=i,analytic=expected,numeric=actual,error=error,passed=passed))
    arrays['latent_score']=score.numpy();fix=fixtures()
    assert digest(q.state_dict())==before and all(v.grad is None for v in q.parameters())
    checks=dict(real_frozen_step_owncarry_IID_identity_and_reset=True,latent_joint_reconstruction=True,
        fixed_exogenous_nonterminal_clock=all_fixed,raw_2time_and_clipped_atom_counterexample=True,numeric3=len(numeric)==3 and all(x['passed'] for x in numeric),
        all_finite=all(bool(np.isfinite(v).all()) for v in arrays.values()),model_unchanged=True)
    result=dict(state='TEMPORAL_LATENT_CONTRACT_PASS' if all(checks.values()) else 'SEMANTIC_GATE_FAILED',checks=checks,metrics=metrics,numeric=numeric,fixtures=fix,
        array_hashes={k:digest(v) for k,v in arrays.items()},raw_sigma=science['raw_sigma'],tau_seconds=2.,
        behavior_hash=digest(dict(weight_policy=p['initial']['policy_hash'],noise='raw_latent_OU_clock_v1',sigma=science['raw_sigma'],tau_seconds=2.)),
        no_native_or_optimizer=True,no_environment_outcome_or_onpolicy_AR_samples=True,no_exploration_superiority_or_learning_admission=True,
        native_worker_clock_integration_NOT_yet_verified=True,conditional_clip_product_NOT_marginal_command_history_density=True)
    return result,arrays

def verify(p):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    Wait.resources();assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    for f,h in p['inputs'].items():assert sha(f)==h,f

def execute(action):
    p=read(OUT/'protocol.json');verify(p);started=time.time();last=[0.]
    def guard():
        assert time.time()<min(started+p['main_seconds' if action=='collect' else 'audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>3:verify(p);last[0]=time.time()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result,arrays=compute(p,False,guard)
    if action=='collect':
        with (OUT/'contract_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result);save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),physics=0,commands=0,gate=result['state'],new_Actor=0,new_Value=0,new_optimizer=0))
    else:
        assert not Path('/proc',str(read(OUT/'summary.json')['pid'])).exists();assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'contract_arrays.npz',allow_pickle=False) as saved:
            assert set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_real_history_sampler_density_and_numeric_arrays_exact=True,new_native=0,new_optimizer=0,gate=result['state']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':execute(sys.argv[1])
