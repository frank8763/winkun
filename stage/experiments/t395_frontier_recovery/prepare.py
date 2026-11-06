"""One registration and zero-native admission, not a navigation result."""
from t395_learning import *
import t395_learning as LOCAL
import ast

def version(job):
    j=copy.deepcopy(job);j['id']='new070_80_'+j['id'];j['case']['step_cap']=40000
    j.update(wide_course=True,circle_shift_x_m=.13000002205371863)
    return j

def prepare():
    assert not OUT.exists()
    storage=ROOT/'stage/runs/t395_storage_reserve'
    assert read(storage/'summary.json')['state']=='COMPLETE' and read(storage/'audit.json')['state']=='PASS'
    prior=ROOT/'stage/runs/t394_takeover_diagnostic';assert read(prior/'audit.json')['state']=='PASS'
    for folder in [PRIOR,prior,storage]:
        assert not Path('/proc',str(read(folder/'summary.json')['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T394.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json';base=read(source)
    centres=[version(j) for j in base['original_jobs']];left=copy.deepcopy(base['narrow_jobs'][0]);left['id']='new070_80_course_LEFT'
    paths=[*HERE.glob('*.py'),source,PARENT,REFERENCE,RECOVERY,PRIOR/'summary.json',PRIOR/'audit.json',prior/'summary.json',prior/'audit.json',receipt,
        storage/'summary.json',storage/'audit.json',DOC/'T395_PLAN.md',DOC/'T395_FRONTIER_METHODS.md',DOC/'T392_PRIMARY_METHODS.md',
        ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',
        ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p=dict(id='T395_ONE_SECOND_FRONTIER_RECOVERY_PG',source_protocol=str(source),inputs={str(f):sha(f) for f in sorted(paths)},
        user_answer='Agree one-second 56-55-54-53-52 frontier training; immutable80s reward; capacity admission first',
        task_version='new070_80_recovery_segment_only',centres=centres,centre_left=left,shift_x_m=.13000002205371863,
        prefix_holds=[280,275,270,265,260],rounds=6,batch_size=4,seed=20261106395,raw_sigma=[.2,.2],lr=.0001,epochs=8,
        ppo_clip=.1,max_logratio=.5,demo_weight=.1,parent_policy_hash=read(PRIOR/'summary.json')['result']['model']['policy_hash'],
        prefix_advance='retained update AND >=2/4 safe new suffixes AND same-prefix deterministic mode safe',
        stop='6 rounds or2 consecutive numeric rejections; shared-contract failure stops with exact ActorAdam rollback',
        final_bytes=12*1024**2,full_bytes=400*1024,max_commands=110,max_physics=4000000,
        main_seconds=5400,learning_seconds=3600,audit_seconds=900,backup_seconds=600,all_seconds=6960,
        old_regression='initial and final old8 measured; no old275 in this recovery-segment pilot',
        full_old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False,
        limitations='Single-seed frontier pilot;52s suffix is not full original six success; full promotion not scheduled',
        deployment='single causal120 recurrent Actor no map/planner/waypoints/runtime teacher/router/mask')
    old8=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')];assert len(old8)==8
    compact_count=8+1+6+5+8+2+2
    worst_bytes=25*p['full_bytes']+7*224*1024+compact_count*5*1024+512*1024
    worst_physics=2*24*40000+2*40000+6*40000+10*40000+2*sum(j['case']['step_cap'] for j in old8)+4*40000+2*28000
    assert worst_bytes<=p['final_bytes'] and worst_physics<=p['max_physics']
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'blocks').mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T395_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],limit=1024**3,
        own=p['final_bytes'],worst_output_bytes=worst_bytes,full24_plus_partial1=25*p['full_bytes'],models7=7*224*1024,
        compact_max=compact_count*5*1024,metadata=512*1024,main_commands_upper=89,preflight_geometry=14,preflight_setup=1,
        max_commands=110,worst_physics=worst_physics,max_physics=p['max_physics'],live48_shadow48_audit48=True,preflight4_discarded_Adam=True))
    print('T395_REGISTERED_ONE_SECOND_RECOVERY_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);parent=Actor(base,p).eval().requires_grad_(False);parent.sample=False
    assert q.phash()==p['parent_policy_hash']
    reference=N.S.arrays(REFERENCE);x=torch.tensor(reference['states']);h=ph=None;sample=[];mode=[]
    q.sample=True;q.rng=np.random.default_rng(395);q.reset_noise()
    with torch.no_grad():
        for row in x:
            a,loc,h,_=q.step(row[None],h);pa,pl,ph,_=parent.step(row[None],ph)
            assert torch.equal(loc,pl) and torch.equal(h,ph)
            sample.append(a[0].numpy());mode.append(pa[0].numpy())
    a={k:copy.deepcopy(v) for k,v in reference.items() if not k.startswith('actor_proposal_')}
    a.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]})
    first=280;a['stochastic'][:first]=0
    assert np.array_equal(np.clip(a['raw_actions'],-1,1).astype(np.float32),np.asarray(sample))
    assert np.allclose(a['raw_actions'],a['locations']+.2*a['innovations'],rtol=0,atol=0)
    _,_,errors=blocks(q,[a]*4,first,p)
    rng=np.random.default_rng(20261106395);z=rng.normal(size=(len(x),128,2))
    exe=np.clip(a['locations'][:,None]+.2*z,-1,1).astype(np.float32);delta=np.abs(exe-np.asarray(mode)[:,None])
    spread=np.quantile(delta,.95,axis=(0,1));assert (spread>.05).all(),spread
    # The PG preflight has deliberately synthetic terminal rewards; never counts
    # as actual environment feedback or a navigation sample.
    synthetic=[]
    for i in range(4):
        aa=copy.deepcopy(a);aa['rewards'][-1]+=.1*i;synthetic.append(aa)
    refs=datasets(q);shadow,so=learner(base,p);before=state(q,opt);short=dict(p,epochs=2)
    live=fit(q,opt,synthetic,first,short,refs,lambda:None,lambda *_:None)
    other=fit(shadow,so,synthetic,first,short,refs,lambda:None,lambda *_:None)
    assert live==other and statehash(q,opt)==statehash(shadow,so)
    load(q,opt,before);assert digest(state(q,opt))==digest(before)
    # Score ignores forced prefix likelihoods, yet suffix recurrent gradient
    # still reaches the prefix observations; future inputs cannot reach past.
    xx=x.clone().requires_grad_(True);b=shadow.base_actions(xx);out=shadow.sequence(xx,b)
    dx=torch.autograd.grad(out[first].sum(),xx)[0]
    assert torch.count_nonzero(dx[:first]) and not torch.count_nonzero(dx[first+1:])
    geom=[];m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);original=copy.deepcopy(m.ACTIVE)
    cancellation_probe=False;signal_probe=False
    try:
        old8=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
        for j in [*p['centres'],*base['original_jobs'],p['centre_left'],*old8]:
            m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            env=obs=None
            try:
                env,obs,_=N.K.R.task_env(create,j);obs=N.K.C.perturb_anchor(env,obs,j)
                assert not env._collision()
                v=N.K.T.L.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,np.float32),0)
                geom.append(dict(id=j['id'],cap=j['case']['step_cap'],scene=N.K.C.model_hash(env),initial=digest(N.K.R.integration(env)),input=digest(v)))
            finally:
                if env is not None:w.events.pop(id(env),None);env.close()
        m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=p['shift_x_m']
        def cancelled(physical=False):
            if physical:raise StageStop('Zero-native cancellation probe')
        ss=Session(p['centre_left'],native,create,w,cancelled,lambda:(_ for _ in ()).throw(AssertionError('No probe physics')),q)
        try:
            initial=ss.initial.copy()
            try:ss.step();raise AssertionError('Cancellation did not stop before physics')
            except StageStop:cancellation_probe=True
            assert ss.steps==0 and not ss.rows and equal(initial,N.K.R.integration(ss.env))
            handler=install_stop_handlers()
            assert signal.getsignal(signal.SIGTERM) is handler
            os.kill(os.getpid(),signal.SIGTERM)
            try:check_stop();raise AssertionError('Pending signal not raised by guard')
            except StageStop:signal_probe=True
            LOCAL.STOP_REQUESTED=None
        finally:ss.close()
    finally:w.close()
    # Max-shape incompressible payload check, preserving all120 observations,
    # hidden, native integration, causal history and mixed likelihood variables.
    padded={};r=np.random.default_rng(392)
    for k,v in N.S.packed(a).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape
        padded[k]=r.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else r.integers(0,100,size=shape,dtype=v.dtype)
    padded['ownership']=r.integers(0,2,size=400,dtype=np.uint8);padded['hold_sha256']=r.integers(0,256,size=(400,32),dtype=np.uint8)
    padded['task_goal_xy']=np.zeros(2);padded['scene_sha256']=np.zeros(32,np.uint8)
    full=len(N.R.blob(padded));assert full<=p['full_bytes'],full
    compact=dict(actions=r.normal(size=(400,2)).astype(np.float32),native_counts=r.integers(0,100,size=400,dtype=np.uint8),end_flags=np.zeros((400,3),np.uint8))
    assert len(N.R.blob(compact))<=5*1024
    model=export(shadow,so,OUT/'preflight_clone.pt.gz');assert Path(model['path']).stat().st_size<=224*1024
    # This generated clone is the preflight stand-in, retained and charged.
    # Its bytes are within the metadata reserve, not a deployed candidate.
    save(OUT/'geometry.json',geom)
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=len(geom)+1,native_setup_commands=1,discarded_Adam=4,
        deterministic_safeT255_commands_exact=True,density_errors=errors,executed_p95_abs_change=spread.tolist(),
        raw_innovation_sigma_exact=True,full400_incompressible=full,synthetic_PG_and_Adam_shadow_rollback=True,
        initial_T393_policy_command_hidden_exact=True,
        prefix_history_gradient_after_readout_update=True,future_causality=True,
        cancel_before_first_physics=cancellation_probe,actual_process_SIGTERM_caught=signal_probe,teacher_actions_not_onpolicy=True))
    print('T395_PREFLIGHT_PASS_ZERO_NATIVE_4_CLONE_ADAM')

if __name__=='__main__':globals()[sys.argv[1]]()
