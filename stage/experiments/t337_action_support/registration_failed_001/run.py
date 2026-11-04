"""Read-only action-distribution feasibility; no native or optimizer imports."""
from pathlib import Path
import sys,json,os,time,hashlib,shutil,signal,traceback,fcntl,subprocess
import numpy as np
import torch
from scipy.special import ndtr,betainc
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t337_action_support';DOC=ROOT/'stage/docs/review_20261104'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file())
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    assert used(ROOT/'stage/runs')-used(OUT)+p['final_bytes']<=p['global_bytes']
    assert used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    assert 'mujoco' not in sys.modules,'This contract cannot create native environments'

def prepare():
    assert not OUT.exists();old=ROOT/'stage/runs/t336_reverse_online'
    assert read(old/'summary.json')['state']=='COMPLETE'
    assert read(ROOT/'stage/docs/overnight_20261103/T336_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T336.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    cases=[json.loads(x) for x in (old/'cases.jsonl').read_text().splitlines()]
    selected=[r for r in cases if r['role']=='training' and r['job_id'].startswith('r0_')]
    assert len(selected)==8 and all(r['full_arrays_saved'] for r in selected)
    files={str(f) for f in HERE.glob('*.py')}|{str(DOC/'RESEARCH_DECISION.md'),str(DOC/'T337_PLAN.md'),str(receipt),str(old/'cases.jsonl'),str(old/'summary.json'),str(old/'protocol.json'),str(ROOT/'stage/docs/overnight_20261103/T336_AUDIT.json')}
    files|={r['path'] for r in selected}
    files|={v['path'] for v in read(old/'summary.json')['models'].values()}
    files.add(str(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'))
    p=dict(experiment='T337_READ_ONLY_BOUNDED_ACTION_SUPPORT',seed=20261104337,draws_per_state=512,
        targets=[.035,.03],bisection_steps=48,quadrature_nodes=256,wall_seconds=180,audit_seconds=180,
        final_bytes=256*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
        physics_steps=0,geometry_commands=0,optimizer_calls=0,selected_cases=selected,
        original_cases_path=str(old/'cases.jsonl'),input_hashes={f:sha(f) for f in sorted(files)})
    assert used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED_READ_ONLY',full_reservation_bytes=used(ROOT/'stage/runs')-used(OUT)+p['final_bytes'])))

def source_arrays(p):
    groups={}
    for arm,expected in [('fixed',674),('reverse',152)]:
        rows=[r for r in p['selected_cases'] if r['arm']==arm];aa=[];mm=[]
        for r in rows:
            assert sha(r['path'])==r['sha256']
            with np.load(r['path']) as a:
                assert a['states'].shape[1]==120 and a['actions'].shape[1]==2
                aa.append(a['actions'].copy());mm.append(a['latent_means'].copy())
        a,m=np.concatenate(aa),np.concatenate(mm);assert len(a)==len(m)==expected
        groups[arm]=(a,m)
    return groups
def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();g=source_arrays(p)
    save(OUT/'preflight.json',dict(state='PASS',source_rows={k:len(v[0]) for k,v in g.items()},
        geometry_commands=0,physics_steps=0,optimizer_calls=0,random_draws=0,
        protocol_sha256=sha(HERE/'protocol.json')))

def clip_moments(mu,sigma):
    zl=(-1-mu)/sigma;zu=(1-mu)/sigma;pl=ndtr(zl);pu=ndtr(-zu);mid=ndtr(zu)-pl
    phil=np.exp(-.5*zl**2)/np.sqrt(2*np.pi);phiu=np.exp(-.5*zu**2)/np.sqrt(2*np.pi)
    mean=-pl+mu*mid+sigma*(phil-phiu)+pu
    second=pl+(mu**2+sigma**2)*mid+2*mu*sigma*(phil-phiu)+sigma**2*(zl*phil-zu*phiu)+pu
    return mean,np.maximum(second-mean**2,0),pl,pu
def calibrated_clip(mode,target,steps):
    lo=np.full_like(mode,1e-6);hi=np.full_like(mode,.25)
    for _ in range(steps):
        mid=(lo+hi)/2;v=clip_moments(mode,mid)[1];low=v<target**2
        lo=np.where(low,mid,lo);hi=np.where(low,hi,mid)
    return (lo+hi)/2
def beta_parameters(mode,target,steps):
    location=(mode+1)/2;lo=np.full_like(mode,1e-6);hi=np.full_like(mode,100000.)
    for _ in range(steps):
        k=(lo+hi)/2;a=1+k*location;b=1+k*(1-location);v=4*a*b/((k+2)**2*(k+3));low=v>target**2
        lo=np.where(low,k,lo);hi=np.where(low,hi,k)
    k=(lo+hi)/2;return 1+k*location,1+k*(1-location),k
def capg_logp(action,mu,sigma):
    density=-.5*((action-mu)/sigma)**2-torch.log(sigma)-.5*np.log(2*np.pi)
    return torch.where(action<=-1,torch.special.log_ndtr((-1-mu)/sigma),
        torch.where(action>=1,torch.special.log_ndtr((mu-1)/sigma),density))
def summarize(draws,mode):
    err=np.abs(draws-mode[:,None,:]);std=draws.std(axis=1)
    return dict(actual_action_std_median=np.median(std,axis=0).tolist(),actual_action_std_min=std.min(axis=0).tolist(),
        absolute_change_median=np.median(err,axis=(0,1)).tolist(),boundary_atom_fraction=(np.abs(draws)==1).mean(axis=(0,1)).tolist(),
        inward_change_at_least_001_fraction=(np.abs(draws-mode[:,None,:])>=.01).mean(axis=(0,1)).tolist())

def compute(p):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(p['seed']);groups=source_arrays(p)
    result=dict(scope='OFFLINE_UNTRAINED_DISTRIBUTION_CONTRACT_NOT_NAVIGATION',rows={},
        new_native_steps=0,new_optimizer_calls=0,trained_models=0,draws_per_state=p['draws_per_state'])
    maxinit=maxsigma=maxbeta=maxnorm=maxscore=maxfd=0.;n=torch.tensor([-1.,-.5,0.,.5,1.],dtype=torch.float64)
    for arm,(actions,latent) in groups.items():
        mode=torch.tanh(torch.tensor(latent)).numpy().astype(np.float64);target=np.asarray(p['targets'])[None,:]
        sigma=calibrated_clip(mode,target,p['bisection_steps']);mean,var,pl,pu=clip_moments(mode,sigma)
        a,b,k=beta_parameters(mode,target,p['bisection_steps']);betamode=2*(a-1)/k-1
        margin=1+sigma;loc=margin*np.tanh(np.arctanh(mode/margin));jac=margin*(1-(mode/margin)**2)
        maxinit=max(maxinit,float(np.max(np.abs(loc-mode))),float(np.max(np.abs(betamode-mode))))
        maxsigma=max(maxsigma,float(np.max(np.abs(np.sqrt(var)-target))))
        betavar=4*a*b/((a+b)**2*(a+b+1));maxbeta=max(maxbeta,float(np.max(np.abs(np.sqrt(betavar)-target))))
        nodes,weights=np.polynomial.legendre.leggauss(p['quadrature_nodes']);x=nodes[None,None,:]
        z=(x-mode[:,:,None])/sigma[:,:,None];pdf=np.exp(-.5*z**2)/(np.sqrt(2*np.pi)*sigma[:,:,None])
        integral=(pdf*weights).sum(-1);norm=pl+pu+integral;maxnorm=max(maxnorm,float(np.max(np.abs(norm-1))))
        zl=(-1-mode)/sigma;zu=(1-mode)/sigma
        dpl=-np.exp(-.5*zl**2)/(np.sqrt(2*np.pi)*sigma);dpu=np.exp(-.5*zu**2)/(np.sqrt(2*np.pi)*sigma)
        score=(pdf*z/sigma[:,:,None]*weights).sum(-1)+dpl+dpu;maxscore=max(maxscore,float(np.max(np.abs(score))))
        # Mixed endpoint masses and interior pdf: finite differences agree with
        # automatic differentiation, including far-tail log probabilities.
        mu=torch.tensor(mode[[0,len(mode)//2,-1]],dtype=torch.float64,requires_grad=True);sig=torch.tensor(sigma[[0,len(mode)//2,-1]])
        for point in n:
            y=torch.full_like(mu,float(point));lp=capg_logp(y,mu,sig);grad=torch.autograd.grad(lp.sum(),mu)[0]
            h=1e-6;fd=(capg_logp(y,mu.detach()+h,sig)-capg_logp(y,mu.detach()-h,sig))/(2*h)
            assert torch.isfinite(lp).all() and torch.isfinite(grad).all()
            maxfd=max(maxfd,float(((grad-fd).abs()/(1+grad.abs())).max()))
        z=rng.normal(size=(len(mode),p['draws_per_state'],2))
        olddraw=np.tanh(latent.astype(np.float64)[:,None,:]+np.exp(-2.3)*z)
        newdraw=np.clip(mode[:,None,:]+sigma[:,None,:]*z,-1,1)
        betadraw=2*rng.beta(a[:,None,:],b[:,None,:],size=z.shape)-1
        threshold=np.where(mode>=0,mode-.01,mode+.01);gauss_prob=np.where(mode>=0,ndtr((threshold-mode)/sigma),ndtr((mode-threshold)/sigma))
        beta_prob=np.where(mode>=0,betainc(a,b,(threshold+1)/2),1-betainc(a,b,(threshold+1)/2))
        result['rows'][arm]=dict(n_states=len(mode),mode_above_999_fraction=(np.abs(mode)>.999).mean(0).tolist(),
            recorded_sample_change_median=np.median(np.abs(actions-mode),axis=0).tolist(),
            tanh_normal=summarize(olddraw,mode),calibrated_CAPG=summarize(newdraw,mode),calibrated_Beta=summarize(betadraw,mode),
            CAPG_expected_action_minus_mode_median=np.median(mean-mode,axis=0).tolist(),
            Beta_expected_action_minus_mode_median=np.median(2*a/(a+b)-1-mode,axis=0).tolist(),
            CAPG_initial_raw_sigma_range=[sigma.min(0).tolist(),sigma.max(0).tolist()],
            new_loc_residual_jacobian_range=[jac.min(0).tolist(),jac.max(0).tolist()],
            old_tanh_residual_jacobian_range=[(1-mode**2).min(0).tolist(),(1-mode**2).max(0).tolist()],
            probability_move_inward_001_median=dict(CAPG=np.median(gauss_prob,axis=0).tolist(),Beta=np.median(beta_prob,axis=0).tolist()))
    checks=dict(initial_deterministic_mode_max_error=maxinit,calibrated_CAPG_std_max_error=maxsigma,
        calibrated_Beta_std_max_error=maxbeta,CAPG_mass_normalization_max_error=maxnorm,
        CAPG_expected_score_max_abs=maxscore,CAPG_autograd_vs_fd_max_relative_error=maxfd)
    result['checks']=checks
    assert maxinit<=1e-12 and maxsigma<=1e-8 and maxbeta<=1e-8 and maxnorm<=1e-8 and maxscore<=1e-7 and maxfd<=1e-5,checks
    cases=[json.loads(x) for x in Path(p['original_cases_path']).read_text().splitlines()];tr=[r for r in cases if r['role']=='training']
    result['T336_physical_accounting']=dict(all_physics=10555201,training_unique_total=sum(r['physics_steps'] for r in tr),
        training_unique_prefix=sum(r['prefix_steps'] for r in tr),training_unique_suffix=sum(r['suffix_steps'] for r in tr),
        suffix_replays_are_not_additional_independent_training=True)
    result['limitations']=['No native effect or trained policy improvement tested.','Distribution draws are not navigation samples.',
        'Matching deterministic initialization does not match stochastic means; inward mean shift is reported.',
        'CAPG variance result does not make PPO clipping unbiased or certify safety.',
        'Float64 prototype; actual worker float32 execution/atom flags still need a future registered native contract.',
        'Beta and CAPG are published methods, not original algorithms.']
    return result

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'summary.json').exists()
    start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T337180s')));signal.alarm(175)
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state='FAILED'
    try:
        save(OUT/'status.json',dict(state='READ_ONLY_NUMERICS',pid=os.getpid(),main_epoch=start))
        result=compute(p);save(OUT/'results.json',result);verify(p);state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),physics_steps=0,optimizer_calls=0))
    finally:signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,physics_steps=0,geometry_commands=0,commands=0,optimizer_calls=0,all_workers_stopped=True,workers_spawned=0,own_bytes=used(OUT),global_bytes=used(ROOT/'stage/runs'))
    save(OUT/'summary.json',s);save(OUT/'status.json',s);print(json.dumps(s))

def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T337_AUDIT.json';assert not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    assert s['state']=='COMPLETE';signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T337audit180s')));signal.alarm(175)
    actual=compute(p);assert actual==read(OUT/'results.json');verify(p);signal.alarm(0)
    save(dest,dict(state='PASS',native_added=0,optimizer_calls=0,same_offline_samples_recomputed_not_independent_data=True,
        all_input_SHA_unchanged=True,all_numerics_exact=True,all_original_processes_absent=True,summary_sha256=sha(OUT/'summary.json')))
    print('T337 independent read-only numerical audit PASS0native0optimizer')

if __name__=='__main__':globals()[sys.argv[1]]()
