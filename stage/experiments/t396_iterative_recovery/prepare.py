"""Register and zero-native preflight for two current-student correction cycles."""
from t396_learning import *
import ast

def prepare():
    assert not OUT.exists()
    storage=ROOT/'stage/runs/t396_cold_reserve'
    assert read(storage/'summary.json')['state']=='COMPLETE' and read(storage/'audit.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T396-COLD.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    template=read(ROOT/'stage/runs/t395_frontier_recovery/protocol.json')
    paths=[*HERE.glob('*.py'),PARENT,REFERENCE,RECOVERY,PRIOR/'summary.json',PRIOR/'audit.json',Path(template['source_protocol']),storage/'summary.json',storage/'audit.json',receipt,DOC/'T396_PLAN.md',DOC/'NEXT_AFTER_T395_SOURCES.md',DOC/'T395_FINDINGS.md',ROOT/'stage/experiments/t395_frontier_recovery/t395_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p={k:copy.deepcopy(template[k]) for k in ['source_protocol','centres','centre_left','shift_x_m','raw_sigma','lr','epochs','ppo_clip','max_logratio','demo_weight','parent_policy_hash','deployment']}
    p.update(id='T396_TWO_ITERATIVE_ACTUAL_STATE_CORRECTION_CYCLES',user_answer='Agree two current-student automatic correction cycles plus fresh closed-loop RL, T393 old8-safe initialization, old8 non-regression acceptance',inputs={str(f):sha(f) for f in sorted(set(paths))},prefix_hold=265,learner_holds=2,teacher_start_hold=267,bridge_holds=5,teacher_sd=[.15,.8],teacher_min_sd=[.04,.15],teacher_generations=2,teacher_max_queries_per_cycle=24,cycles=2,batch_size=4,seed=20261106396,final_bytes=9*1024**2,full_bytes=400*1024,search_bytes=384*1024,max_commands=200,max_physics=4000000,main_seconds=5400,learning_seconds=3600,audit_seconds=900,backup_seconds=600,all_seconds=7140,source_seconds=180,auxiliary_weights=dict(original=.25,prior_recovery=.25,current_cumulative_short_bridges=.5),acceptance='numeric AND original old8 all8safe; otherwise exact wholeActorAdam rollback and stop learning',stop='2 cycles, teacher no verified short bridge, candidate rejection, numeric/shared evidence failure or absolute deadline; no restart or grids',matched_control=False,limitations='Single-arm feasibility pilot, not causal benefit or original algorithm claim; no full6 gate scheduled',old275=None,main54=None,fresh16=None,reload78=None,seven_eight=False)
    base=N.science(p);old=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
    assert len(old)==8
    worstbytes=9*p['full_bytes']+4*p['full_bytes']+2*p['search_bytes']+4*224*1024+80*5*1024+768*1024
    worstphys=16*40000+2*(26700+48*13300+4*13300)+4*sum(j['case']['step_cap'] for j in old)+10*40000+4*40000
    assert worstbytes<=p['final_bytes'] and worstphys<=p['max_physics']
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T396_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,reserved=used+p['final_bytes'],own=p['final_bytes'],worst_output_bytes=worstbytes,full_train8_plus_partial1=9*p['full_bytes'],teacher4_full=4*p['full_bytes'],search2=2*p['search_bytes'],models4=4*224*1024,compact80=80*5*1024,metadata=768*1024,main_commands_worst=169,preflight_zero_physics_commands=15,worst_physics=worstphys,max_physics=p['max_physics'],live16_shadow16_audit16=True,preflight_discarded_Adam4=True,matched_control_not_run_capacity_limited=True))
    print('T396_REGISTERED_NOT_LAUNCHED')

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
    first=265;a['stochastic'][:first]=0
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
    refs=groups(q,p,[REFERENCE]);shadow,so=learner(base,p);before=state(q,opt);short=dict(p,epochs=2)
    def event(which):
        def emit(where,k):
            with (OUT/'preflight_optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,where,k])+'\n')
        return emit
    live=fit(q,opt,synthetic,first,short,refs,lambda:None,event('live'))
    other=fit(shadow,so,synthetic,first,short,refs,lambda:None,event('shadow'))
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
            clear_stop()
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
    search=teacher_buffer(p)
    for k,v in search.items():
        if np.issubdtype(v.dtype,np.floating):search[k]=r.normal(size=v.shape).astype(v.dtype)
        else:search[k]=r.integers(0,100,size=v.shape,dtype=v.dtype)
    search_bytes=len(N.R.blob(search));assert search_bytes<=p['search_bytes']
    assert len(teacher_candidates(np.zeros((16,2)),np.ones((16,2)),None,0,0,p))==12
    assert teacher_controls(np.zeros((16,2)),reference['actions'],p).shape==(133,2)
    save(OUT/'geometry.json',geom)
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=len(geom)+1,native_setup_commands=1,all_zero_physics_commands=len(geom)+2,discarded_Adam=4,search_incompressible=search_bytes,
        deterministic_parent_T393_commands_hidden_exact=True,density_errors=errors,executed_p95_abs_change=spread.tolist(),
        raw_innovation_sigma_exact=True,full400_incompressible=full,synthetic_PG_and_Adam_shadow_rollback=True,
        initial_T393_policy_command_hidden_exact=True,
        prefix_history_gradient_after_readout_update=True,future_causality=True,
        cancel_before_first_physics=cancellation_probe,actual_process_SIGTERM_caught=signal_probe,teacher_actions_not_onpolicy=True))
    print('T396_PREFLIGHT_PASS_ZERO_NATIVE_4_CLONE_ADAM')

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            f=OUT/'preflight_optimizer_calls.jsonl';events=[json.loads(v) for v in f.read_text().splitlines()] if f.exists() else []
            save(OUT/'entry_failure.json',dict(mode=sys.argv[1],error=traceback.format_exc(),native=0,completed_preflight_discarded_Adam=sum(v[1]=='AFTER' for v in events)))
        raise
