"""Register complete worst-case budgets and test the actual native/learner seams."""
from t404_core import *
import ast,dis,builtins,types

def prepare():
    assert not OUT.exists();src=read(ROOT/'stage/runs/t393_safe_demo_pg/protocol.json');past=read(PRIOR/'summary.json');assert past['state']=='COMPLETE' and read(PRIOR/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/runs/t403_cold_exact_reserve/audit.json')['state']=='PASS'
    for file in ['launch.json','driver_started.json','main_started.json']:assert not Path('/proc',str(read(PRIOR/file)['pid'])).exists()
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T402R.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    oldrows=[json.loads(x) for x in (PRIOR/'cases.jsonl').read_text().splitlines() if json.loads(x)['role'].startswith('own_old_')];assert len(oldrows)==8
    paths=[*HERE.glob('*.py'),PARENT,REFERENCE,L.RECOVERY,PRIOR/'protocol.json',PRIOR/'summary.json',PRIOR/'audit.json',receipt,ROOT/'stage/docs/review_20261107/T404_PLAN.md',ROOT/'stage/docs/review_20261107/RECOVERY_PRIMARY_SOURCES.md',*[Path(r['path']) for r in oldrows],ROOT/'stage/experiments/t398_warm_recovery_rl/t398_learning.py',ROOT/'stage/experiments/t396_iterative_recovery/t396_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t387_full_horizon_night/night_common.py',ROOT/'stage/experiments/t381_native_mppi_teacher/run.py',Path(src['source_protocol']),DOC/'T404_REGISTRATION_FAILED_001.json']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p={k:copy.deepcopy(src[k]) for k in ['source_protocol','main_jobs','fresh_jobs','reload_jobs','centres','centre_left','shift_x_m','raw_sigma','lr','epochs','ppo_clip','max_logratio','demo_weight','deployment']}
    p.update(id='T404_LAST_AUTHORIZED_NARROW_ATTEMPT',inputs={str(f):sha(f) for f in sorted(set(paths))},user_authorization='2026-11-07 explicit final bounded narrow attempt; if complete6m gate fails pause narrow and switch wide; algorithm/scene/engine changes allowed; real-robot feasibility and scientific rigor required',seed=20261107404,cycles=2,warm_steps=200,consolidation_steps=200,old_weight=1.,teacher_sd=[.15,.8],teacher_min_sd=[.04,.15],teacher_queries_per_cycle=24,deviation_m=.05,deviation_deg=5.,before_contact_holds=10,goal_prefix_hold=260,old_history_files=[r['path'] for r in oldrows],old_support_semantics='Frozen T393 factual old8 histories with initial T402R raw-location targets, not current-policy visitation or recovery expert',expected_origin_digest=past['result']['main_centres'][0]['array_digest'],expected_mode52_digest=past['rounds'][-1]['course_mode']['array_digest'],final_bytes=14*1024**2,wide_reserved_bytes=2*1024**2,full_bytes=400*1024,search_bytes=768*1024,max_commands=900,max_physics=17000000,main_seconds=7200,learning_seconds=4500,audit_seconds=1200,backup_seconds=900,wide_seconds=1200,all_seconds=10800,full_six_gate='same model old275+main54 allGOAL then16unusedfresh then78serialized full-payload native digests exact; audit required',stop='at most2 adaptive teacher/acquire/consolidate/RL cycles; no teacher source or supervised old8/53 rejection stops learning; no restart/grid or additional narrow budget',next_if_failed='predeclared T405 frozen width1.17/80 admission, no long-distance promotion from centres',sim2real='unchanged body/tyres/collision masks/wheel5/force120/actual120 causal input and recurrent actor; teacher privileged state train-only; hardware observation provenance and actuator calibration still required')
    base=read(p['source_protocol']);old8=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
    worst_physics=2*(40000+48*40000+2*40000)+4*40000+16*40000+6*sum(j['case']['step_cap'] for j in old8)+8*40000+sum(j['case']['step_cap'] for j in [*p['centres'],*p['centres'],*p['main_jobs'],*base['final_old_jobs'],*p['fresh_jobs'],*p['reload_jobs']])
    worst_bytes=18*p['full_bytes']+2*p['search_bytes']+9*224*1024+450*5*1024+1024**2
    assert worst_physics<=p['max_physics'] and worst_bytes<=p['final_bytes'];used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']+p['wide_reserved_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    ids=sorted(j['id'] for j in p['fresh_jobs']);ledger={};consumed=[]
    for f in sorted((ROOT/'stage/runs').rglob('*.jsonl')):
        if f.name not in ['cases.jsonl','replays.jsonl','evaluations.jsonl']:continue
        ledger[str(f)]=sha(f)
        for line in f.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if not isinstance(r,dict):continue
            keys=[r.get(k,'') for k in ['id','job_id','role','task_id']]
            if any(isinstance(v,str) and any(j==v or v.endswith('_'+j) for j in ids) for v in keys):consumed.append(str(f))
    assert not consumed,consumed
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();(OUT/'teachers').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(OUT/'fresh_admission.json',dict(state='PASS_NO_PRIOR_NEW_VERSION_NAVIGATION',job_ids=ids,locked_ledger_SHA=ledger,consumed=consumed))
    save(DOC/'T404_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,global_full_worst=used+p['final_bytes']+p['wide_reserved_bytes'],global_limit=2**30,own=p['final_bytes'],conditional_wide=p['wide_reserved_bytes'],worst_bytes=worst_bytes,full_slots18_includes4_worker_partial_plus1_main_partial=True,compact_slots=450,model_slots=9,search_slots=2,worst_physics=worst_physics,max_physics=p['max_physics'],max_commands=p['max_commands'],max_live_Adam=816,max_shadow_Adam=816,max_discarded_audit_Adam=816,preflight_discarded_Adam=12))
    print('T404_REGISTERED_NOT_STARTED',worst_physics,worst_bytes,flush=True)

def preflight():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import importlib.util
    spec=importlib.util.spec_from_file_location("stage404_actual_main_preflight",HERE/"run.py");R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    def names(code):
        missing={x.argval for x in dis.get_instructions(code) if x.opname=='LOAD_GLOBAL' and x.argval not in R.__dict__ and not hasattr(builtins,x.argval)}
        for c in code.co_consts:
            if isinstance(c,types.CodeType):missing.update(names(c))
        return missing
    assert not names(R.main.__code__),names(R.main.__code__)
    lock=(OUT/'preflight_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    q,opt=learner(base,p);fixed=digest(q.state_dict());a={k:v for k,v in N.S.arrays(REFERENCE).items() if not k.startswith('actor_proposal_')};x=torch.tensor(a['states']);h=None;q.reset_noise();q.sample=True;q.rng=np.random.default_rng(404);q.tape=[]
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    aa=copy.deepcopy(a);aa.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});aa['stochastic'][:12]=0
    synthetic=[]
    for i in range(4):b=copy.deepcopy(aa);b['rewards'][-1]+=.1*i;synthetic.append(b)
    old=old_groups(q,p);refs=L.datasets(q)
    for g in refs:g.update(weight=.5,stop=len(g['x']))
    shadow,so=learner(base,p);before=state(q,opt);short=dict(p,epochs=2,consolidation_steps=2,warm_steps=2)
    u=L.warm_fit(q,opt,refs,old,short,lambda:None,lambda *_:None,2);v=L.warm_fit(shadow,so,refs,old,short,lambda:None,lambda *_:None,2);assert u==v and L.statehash(q,opt)==L.statehash(shadow,so)
    L.load(q,opt,before);L.load(shadow,so,before)
    new=[dict(x=g['x'],b=g['b'],target=g['y'],first=g['first']) for g in refs]
    u=consolidate(q,opt,old,new,short,lambda:None,lambda *_:None);v=consolidate(shadow,so,old,new,short,lambda:None,lambda *_:None);assert u==v and L.statehash(q,opt)==L.statehash(shadow,so)
    L.load(q,opt,before);L.load(shadow,so,before)
    u=L.fit(q,opt,synthetic,12,short,refs,old,lambda:None,lambda *_:None);v=L.fit(shadow,so,synthetic,12,short,refs,old,lambda:None,lambda *_:None);assert u==v and L.statehash(q,opt)==L.statehash(shadow,so)
    mature=state(q,opt);L.load(q,opt,before);assert digest(state(q,opt))==digest(before);L.load(q,opt,mature);L.load(q,opt,before);assert digest(q.state_dict())==fixed
    xx=x.clone().requires_grad_(True);y=q.sequence(xx,q.base_actions(xx));dx=torch.autograd.grad(y[12].sum(),xx)[0];assert torch.count_nonzero(dx[:12]) and not torch.count_nonzero(dx[13:])
    geom=[];m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);original=copy.deepcopy(m.ACTIVE)
    try:
        old8=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
        for j in [*p['main_jobs'],*p['fresh_jobs'],*old8,p['centre_left']]:
            m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m']);env,obs,_=N.K.R.task_env(create,j)
            try:obs=N.K.C.perturb_anchor(env,obs,j);assert not env._collision();geom.append(dict(id=j['id'],scene=N.K.C.model_hash(env),initial=digest(N.K.R.integration(env))))
            finally:w.events.pop(id(env),None);env.close()
        def stop(physical=False):
            if physical:raise L.StageStop('zero native cancellation')
        s=L.Session(p['centre_left'],native,create,w,stop,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')),q)
        try:
            snap=s.capture();s.restore(snap)
            try:s.step();raise AssertionError('Cancellation failed')
            except L.StageStop:pass
            assert not s.rows and not s.steps
        finally:s.close()
    finally:w.close()
    rng=np.random.default_rng(404);padded={}
    for k,v in N.S.packed(aa).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape
        padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    padded.update(ownership=rng.integers(0,2,400,np.uint8),hold_sha256=rng.integers(0,256,(400,32),np.uint8),task_goal_xy=np.zeros(2),scene_sha256=np.zeros(32,np.uint8))
    assert len(N.R.blob(padded))<=p['full_bytes']
    data=search_buffer(0)
    for v in data.values():
        if np.issubdtype(v.dtype,np.floating):v[:]=rng.normal(size=v.shape)
        else:v[:]=rng.integers(0,100,size=v.shape,dtype=v.dtype)
    assert len(N.R.blob(data))<=p['search_bytes']
    export(q,opt,OUT/'preflight_clone.pt.gz');counts=mp.get_context('spawn').RawArray('q',5);pp=dict(p,native_stop_epoch=time.time()+180);pool=Pool(pp,counts,lambda:None)
    try:
        results=pool.request([dict(model=str(PARENT),job=p['centre_left'],first=0,seed=None,cancel_probe=True)]*4);assert all(v['state']=='CANCEL_PASS' for v in results) and sum(counts)==0
    finally:pool.close()
    save(OUT/'geometry.json',geom);save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=12,zero_physics_commands=len(geom)+1+1+4+4+p.get('prior_preflight_zero_physics_commands',0),prior_discarded_Adam=p.get('prior_preflight_discarded_Adam',0),recursive_main_globals=True,actual_writer_lock=True,mature_Adam_rollback=True,all3_update_paths_exact=True,real_four_workers_READY_cancel_exit=True,full_and_search_incompressible_bounds=True,history_BPTT_causal=True));N.manifest(OUT);print('T404_PREFLIGHT_PASS_ZERO_NATIVE_12_DISCARDED_ADAM',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
