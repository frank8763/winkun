"""One registration and zero-native admission, not a navigation result."""
from t392_learning import *
import ast

def version(job):
    j=copy.deepcopy(job);j['id']='new070_80_'+j['id'];j['case']['step_cap']=40000
    j.update(wide_course=True,circle_shift_x_m=.13000002205371863)
    return j

def prepare():
    assert not OUT.exists()
    previous=ROOT/'stage/runs/t391_actual_student_recovery'
    assert read(previous/'student/audit.json')['state']=='PASS'
    assert not Path('/proc',str(read(previous/'student/summary.json')['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T391.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source=ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json';base=read(source)
    jobs=[version(v) for v in base['six_jobs']];fresh=[version(v) for v in base['fresh_jobs']]
    centres=[copy.deepcopy(j) for j in jobs if j['id'].endswith('_13')]
    assert len(jobs)==54 and len(fresh)==16 and len(centres)==2
    left=copy.deepcopy(base['narrow_jobs'][0]);left['id']='new070_80_course_LEFT'
    reload=[version(v) if v['id'].startswith(('dev6_','fresh_hold6_')) else copy.deepcopy(v) for v in base['reload_jobs']]
    paths=[*HERE.glob('*.py'),source,PARENT,REFERENCE,RECOVERY,previous/'student/audit.json',receipt,
        DOC/'T392_PLAN.md',DOC/'T392_PRIMARY_METHODS.md',
        ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p=dict(id='T392_DEMO_ASSISTED_TRUE_CLOSED_LOOP_PG',source_protocol=str(source),
        inputs={str(f):sha(f) for f in sorted(paths)},user_answer='Agree: 0.70m gap / 80s main task',
        task_version='new070_80_same_scene_both_sides',original_task='057m_56s_diagnostic_only',
        main_jobs=jobs,fresh_jobs=fresh,reload_jobs=reload,centres=centres,centre_left=left,
        shift_x_m=.13000002205371863,prefix_holds=[250,175,100,0],rounds=6,batch_size=4,
        seed=20261106392,raw_sigma=[.2,.2],lr=.0001,epochs=8,ppo_clip=.1,max_logratio=.5,demo_weight=.1,
        prefix_advance='retained numeric update AND >=2/4 safe exploration AND deterministic same-prefix suffix GOAL',
        stop='6 rounds or 2 consecutive numeric rejected proposals; censored/incomplete batch stops with rollback',
        final_bytes=int(15.5*1024**2),full_bytes=400*1024,max_commands=1000,max_physics=32000000,
        main_seconds=7200,learning_seconds=3600,audit_seconds=1200,backup_seconds=600,all_seconds=9120,
        retain='all candidate Actor+Adam snapshots; reject restores whole Actor+Adam; exact shadow and independent raw reconstruction',
        sample_semantics='independent SeedSequence([master,round,peer]) streams; exact native repeats excluded from N=4',
        deployment='single causal120 true recurrent Actor; no map/planner/waypoints/runtime teacher/router/mask',
        old_regression='initial old275 measured, final all275 required for promotion; no per-update old20 acceptance',
        promotion='same candidate: both main centres+54 main+275 old all safe, then 16 unconsumed new-version holdouts, then 78 exact native model readbacks',
        limitation='multi-factor engineering pilot; not original DAPG/RFCL or general unbiasedness/safety guarantee; 7/8 not implemented')
    max_eval_jobs=275+2+6+2+2+54+275+16
    max_physics=2*24*40000+6*40000+2*sum(j['case']['step_cap'] for j in base['final_old_jobs'])+sum(j['case']['step_cap'] for j in [*centres,*centres,*centres,*jobs,*fresh,*reload,*base['original_jobs']])
    worst_bytes=25*p['full_bytes']+7*224*1024+max_eval_jobs*5*1024+1024**2
    assert max_physics<=p['max_physics'] and worst_bytes<=p['final_bytes']
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'blocks').mkdir();(OUT/'episodes').mkdir()
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T392_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],limit=1024**3,
        own=p['final_bytes'],worst_output_bytes=worst_bytes,full24_plus_partial1=p['full_bytes']*25,
        models_final_plus6=7*224*1024,compact_max=max_eval_jobs*5*1024,metadata=1024**2,
        worst_physics=max_physics,main_commands_upper=861,preflight_geometry=73,all_command_limit=1000,
        max_new_samples=24,max_exact_replays=24,live_Adam=48,shadow_Adam=48,audit_discarded_Adam=48,
        preflight_discarded_Adam=4,main_wall=7200,audit_wall=1200,backup_wall=600,all_wall=9120))
    print('T392_REGISTERED_NEW070_80_MAIN_NOT_STARTED')

def preflight():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);parent=N.S.NightActor(base,0.).eval().requires_grad_(False);parent.load_learned(N.S.read_model(PARENT));parent.sample=False
    reference=N.S.arrays(REFERENCE);x=torch.tensor(reference['states']);h=ph=None;sample=[];mode=[]
    q.sample=True;q.rng=np.random.default_rng(392);q.reset_noise();parent.reset_noise()
    with torch.no_grad():
        for row in x:
            a,loc,h,_=q.step(row[None],h);pa,pl,ph,_=parent.step(row[None],ph)
            assert torch.equal(loc,pl) and torch.equal(h,ph)
            sample.append(a[0].numpy());mode.append(pa[0].numpy())
    a={k:copy.deepcopy(v) for k,v in reference.items() if not k.startswith('actor_proposal_')}
    a.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]})
    first=250;a['stochastic'][:first]=0
    assert np.array_equal(np.clip(a['raw_actions'],-1,1).astype(np.float32),np.asarray(sample))
    assert np.allclose(a['raw_actions'],a['locations']+.2*a['innovations'],rtol=0,atol=0)
    _,_,errors=blocks(q,[a]*4,first,p)
    rng=np.random.default_rng(20261106392);z=rng.normal(size=(len(x),128,2))
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
    xx=x.clone().requires_grad_(True);b=q.base_actions(xx);out=q.sequence(xx,b)
    dx=torch.autograd.grad(out[first].sum(),xx)[0]
    assert torch.count_nonzero(dx[:first]) and not torch.count_nonzero(dx[first+1:])
    geom=[];m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);original=copy.deepcopy(m.ACTIVE)
    try:
        for j in [*p['main_jobs'],*p['fresh_jobs'],*base['original_jobs'],p['centre_left']]:
            m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            env=obs=None
            try:
                env,obs,_=N.K.R.task_env(create,j);obs=N.K.C.perturb_anchor(env,obs,j)
                assert not env._collision()
                v=N.K.T.L.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,np.float32),0)
                geom.append(dict(id=j['id'],cap=j['case']['step_cap'],scene=N.K.C.model_hash(env),initial=digest(N.K.R.integration(env)),input=digest(v)))
            finally:
                if env is not None:w.events.pop(id(env),None);env.close()
    finally:w.close()
    # Max-shape incompressible payload check, preserving all120 observations,
    # hidden, native integration, causal history and mixed likelihood variables.
    padded={};r=np.random.default_rng(392)
    for k,v in N.S.packed(a).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape
        padded[k]=r.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else r.integers(0,100,size=shape,dtype=v.dtype)
    padded['ownership']=r.integers(0,2,size=400,dtype=np.uint8);padded['hold_sha256']=r.integers(0,256,size=(400,32),dtype=np.uint8)
    full=len(N.R.blob(padded));assert full<=p['full_bytes'],full
    compact=dict(actions=r.normal(size=(400,2)).astype(np.float32),native_counts=r.integers(0,100,size=400,dtype=np.uint8),end_flags=np.zeros((400,3),np.uint8))
    assert len(N.R.blob(compact))<=5*1024
    model=export(shadow,so,OUT/'preflight_clone.pt.gz');assert Path(model['path']).stat().st_size<=224*1024
    # This generated clone is the preflight stand-in, retained and charged.
    # Its bytes are within the metadata reserve, not a deployed candidate.
    save(OUT/'geometry.json',geom)
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=len(geom),discarded_Adam=4,
        deterministic_parent_location_hidden_exact=True,density_errors=errors,executed_p95_abs_change=spread.tolist(),
        raw_innovation_sigma_exact=True,full400_incompressible=full,synthetic_PG_and_Adam_shadow_rollback=True,
        full_prefix_history_gradient_and_future_causality=True,teacher_actions_not_onpolicy=True))
    print('T392_PREFLIGHT_PASS_ZERO_NATIVE_4_CLONE_ADAM')

if __name__=='__main__':globals()[sys.argv[1]]()
