"""Bounded preparation; native registration requires explicit current window."""
from pathlib import Path
import copy,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import t555_runtime as R
L,ROOT,OUT,DOC=R.L,R.ROOT,R.OUT,R.DOC
torch,np,read,save,sha,digest=L.torch,L.np,L.read,L.save,L.sha,L.digest
PRIOR=ROOT/'stage/runs/t552_joint_robust_night'

def configuration():
    base=read(PRIOR/'protocol.json');p=read(base['actor_configuration']);p.update(base)
    p.update(retention_zero_norm=1e-12,arm='project')
    return p

def preflight():
    DOC.mkdir(parents=True,exist_ok=True);assert not (DOC/'PREFLIGHT.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15}
    p=configuration();rng=np.random.default_rng(555);tests=[]
    for name,u,h in [('zero',[0.,0.],[0.,0.]),('aligned',[1.,0.],[1.,0.]),('safe',[-1.,0.],[1.,0.]),('orthogonal',[0.,1.],[1.,0.]),('tiny',[1.,2.],[1e-17,0.])]+[(str(i),rng.normal(size=7).tolist(),rng.normal(size=7).tolist()) for i in range(12)]:
        u,h=torch.tensor(u,dtype=torch.float64),torch.tensor(h,dtype=torch.float64)
        proj,scalar,m=L.directions(u,h);assert torch.isfinite(proj).all()
        if m['active']:
            assert float(h@proj)<=1e-12*max(float(h.norm()*u.norm()),1e-30)
            # Orthogonal Euclidean projection obeys Pythagoras.
            assert abs(float(u.square().sum()-proj.square().sum()-(u-proj).square().sum()))<1e-12
        tests.append(dict(case=name,**m))
    traces=[];steps=0
    def event(*args):
        nonlocal steps
        if args[0].endswith('_AFTER'):steps+=1
        save(DOC/'PREFLIGHT_PROGRESS.json',dict(discarded_Adam_completed=steps,event=list(args),native=0,epoch=time.time()))
    for arm in ['project','scalar']:
        cfg=dict(p,arm=arm);obj=L.make(cfg);shadow=L.make(cfg);data,pairs=L.cached_dataset(cfg,limit=12);before=L.state(obj)
        a=L.bc(obj,data,pairs,cfg,lambda:None,event,steps=2);b=L.bc(shadow,data,pairs,cfg,lambda:None,event,steps=2)
        assert a==b and digest(L.state(obj))==digest(L.state(shadow));mature=L.state(obj);arrays=[]
        for seed in range(4):
            a=L.G.E.N.S.arrays(cfg['keep_rows'][0]['path']);n=len(a['states']);a={k:v[:8].copy() if getattr(v,'ndim',0)>0 and len(v)==n else v.copy() for k,v in a.items()}
            q=copy.deepcopy(obj[0]).requires_grad_(False);q.sample=True;q.reset_noise();q.rng=np.random.default_rng(5550+seed);h=None
            with torch.no_grad():
                for x in a['states']:u,loc,h,_=q.step(torch.tensor(x[None]),h)
            for name in ['locations','logp','raw_actions','atom']:a[name]=np.asarray([t[name] for t in q.tape])
            a['ownership']=np.zeros(8,np.uint8);a['stochastic']=np.ones(8,np.uint8);a['end_flags']=np.zeros((8,3),np.uint8);a['end_flags'][-1,2]=1;arrays.append(a)
        a=L.rl_or_control(obj,arrays,data,pairs,dict(cfg,rl_steps=2),lambda:None,event);b=L.rl_or_control(shadow,arrays,data,pairs,dict(cfg,rl_steps=2),lambda:None,event)
        assert a==b and digest(L.state(obj))==digest(L.state(shadow));L.load(obj,mature);assert digest(L.state(obj))==digest(mature)
        try:raise InterruptedError('synthetic interruption after mature optimizer')
        except InterruptedError:L.load(obj,before)
        assert digest(L.state(obj))==digest(before)
        traces.append(dict(arm=arm,synthetic_density_and_shadow=True,mature_and_interrupt_rollback=True))
    assert steps==16
    # A single common saved mature Adam state: projection versus locally equal
    # length control. Two discarded updates, not new online navigation learning.
    s=read(PRIOR/'round_01_rl/summary.json')['result'];cfg=dict(p,initial=s['behavior']);data,pairs=L.cached_dataset(cfg)
    arrays=[L.G.E.N.S.arrays(x['path']) for x in R.G.rows(PRIOR/'round_01_rl')[:4]];finite=[];states=[]
    for arm in ['project','scalar']:
        z=dict(cfg,arm=arm,rl_steps=1);obj=L.make(z);before_values=L.objectives(obj[0],data,z)[1]
        u=L.rl_or_control(obj,arrays,data,pairs,z,lambda:None,event);after_values=L.objectives(obj[0],data,z)[1]
        finite.append(dict(arm=arm,before=before_values,after=after_values,update=u));states.append(L.state(obj))
    assert steps==18 and digest(states[0]['actor_opt'])==digest(states[1]['actor_opt'])
    ma,mb=[x['update']['trace'][0]['displacement'] for x in finite]
    assert ma['active'] and mb['active'] and abs(ma['applied_norm']-mb['applied_norm'])<1e-12
    assert ma['applied_retention_dot']<=1e-16 and mb['applied_retention_dot']>0
    assert finite[0]['after']['keep']+finite[0]['after']['neighbor']<finite[1]['after']['keep']+finite[1]['after']['neighbor']
    # Packed history bound from actual schema, extrapolated to declared longest
    # training episode. Arrays are RAM-only here, 0 new physical transitions.
    a=arrays[0];packed=R.E.N.S.packed(a);n=len(a['states']);maxn=max((j['case']['step_cap']+99)//100 for j in cfg['training_jobs'])
    bound=65536
    for v in packed.values():
        bound+=maxn*int(v.nbytes/v.shape[0]) if v.ndim and v.shape[0]==n else v.nbytes
    # Twelve decision fields/history fields and innovation fields are already
    # present in source packed schema; zip overhead reserved above.
    assert bound<cfg['full_bytes']
    src={str(f):sha(f) for f in HERE.glob('*.py')}
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',epoch=time.time(),pid=os.getpid(),new_native=0,new_navigation_updates=0,synthetic_discarded_Adam=16,saved_data_discarded_Adam=2,total_discarded_Adam=steps,geometry_tests=tests,contracts=traces,same_mature_state_finite_step=finite,real_same_state_step_norm_matched=True,moments_equal=True,full_payload_uncompressed_upper_bytes=bound,source=src,not_closed_loop_success=True))
    print('T555_PREFLIGHT_PASS',steps,bound,flush=True)

def prepare():
    assert not OUT.exists();auth=read(DOC/'AUTHORIZATION.json');pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE'
    assert all(sha(f)==h for f,h in pref['source'].items());p=read(PRIOR/'protocol.json')
    p.update(id='T555_ACTUAL_ADAM_RETENTION_DIRECTION_VS_LOCAL_LENGTH_CONTROL',arms=['project','scalar'],max_rounds_per_arm=1,retention_zero_norm=1e-12,max_bytes=int(22.5*2**20),child_reserve_bytes=7*2**20,child_bytes=7*2**20,final_bytes=8*2**20,main_seconds=2400,learning_seconds=1200,audit_seconds=600,final_reserve_seconds=7200,transport_seconds=600,source_seconds=180,max_physics=auth['max_physics'],max_commands=auth['max_commands'],max_main_seconds=auth['max_main_seconds'],absolute_learning_stop=auth['learning_stop_epoch'],absolute_native_stop=auth['native_stop_epoch'],absolute_all_stop=auth['all_stop_epoch'],training_seeds_fixed=[552100,552101,552102,552103],scope='Twofixedone-roundarms:postAdamaverage-retentionhalfspacevsownstateequal-displacement-lengthordinarydirection.32BC+4freshPPOmaximum,5986readout,T550fixed,oldloss/data/reward/sigmaunchanged. Momentsretainrawsumgradient,entireActorAdamrollbackonreject. ZeroMSEgradientcannotprotectfirststep;finite-step/closedloopsnotguaranteed;meanretentionnotperID. Eachdifferentbehaviorcollectsownfresh4and4exact;38plus38exact. BaselineT552historicalNOTnewindependentdata. Full407onlyacceptedselectedcandidate;unchangedbaseline407reused. NoCAGrad/drift/capacity/teacher/seedgrid/newlayoutadded. No7/8newgate/promotion/freshclaim.')
    p['inputs']={f:h for f,h in p['inputs'].items()}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
    for f in [DOC/'AUTHORIZATION.json',DOC/'PLAN.md',DOC/'PREFLIGHT.json',ROOT/'stage/docs/day_20261115/CONFLICT_PILOT_SOURCE_CHECK.md',ROOT/'stage/runs/t554_day_conflict_probe/diagnostic.json']:p['inputs'][str(f)]=sha(f)
    roundphys=2*sum(j['case']['step_cap'] for j in p['training_jobs']+p['evaluation_jobs']);finalphys=sum(j['case']['step_cap'] for j in p['final_jobs'])
    assert 2*roundphys+finalphys+2000<=p['max_physics'] and 2*88+411+8<=p['max_commands']
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    jobs=copy.deepcopy(p['training_jobs'][:2])
    for j in jobs:j['case']['step_cap']=500
    f=OUT/'native_contract';f.mkdir();save(f/'protocol.json',dict(kind='engineering',initial=p['initial'],probe_jobs=jobs,probe_seeds=[5550,5551],max_physics=2000,max_commands=8,main_seconds=300,learning_seconds=0,child_bytes=512*1024,_root_sha256=sha(OUT/'protocol.json')))
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=2*roundphys+finalphys+2000,max_commands=595,round_physics=roundphys,final_physics=finalphys,max_main_seconds=2*2400+7200+300,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,all_stop=p['absolute_all_stop'],native_contract_worst=2000,preflight_discarded_Adam=18));print('T555_REGISTERED')

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        DOC.mkdir(parents=True,exist_ok=True)
        save(DOC/('ERROR_'+sys.argv[1]+'_'+str(int(time.time()))+'.json'),dict(error=traceback.format_exc(),new_native_launch=False,epoch=time.time()));raise
