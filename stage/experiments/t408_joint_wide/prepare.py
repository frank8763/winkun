"""Freeze the joint-support pilot, then exercise its actual zero-native paths."""
from t408_core import *
import ast,importlib.util,types,dis,builtins
def prepare():
    assert not OUT.exists()
    for name in ['t404_last_narrow','t405_wide_admission','t406_wide_full_route','t407_cold_exact_reserve']:
        directory=ROOT/'stage/runs'/name;s=read(directory/'summary.json');assert s['state']=='COMPLETE' and read(directory/'audit.json')['state']=='PASS';assert not Path('/proc',str(s['pid'])).exists()
    src=read(ROOT/'stage/runs/t406_wide_full_route/protocol.json');parent=read(ROOT/'stage/runs/t404_last_narrow/protocol.json');oldrows=[json.loads(x) for x in (ROOT/'stage/runs/t404_last_narrow/cases.jsonl').read_text().splitlines() if x]
    base=read(src['source_protocol']);old8=[copy.deepcopy(j) for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')];assert len(old8)==8
    centres=[copy.deepcopy(j) for j in src['main_jobs'] if j['id'].endswith('_13')];assert len(centres)==2
    jobs=copy.deepcopy(src['main_jobs']);fresh=copy.deepcopy(src['fresh_jobs']);reload=copy.deepcopy(src['reload_jobs']);left=copy.deepcopy(src['wide5_jobs']);right=next(j for j in centres if j['case']['side']=='RIGHT')
    paths=[*HERE.glob('*.py'),PARENT,*map(Path,src['wide_history_files']),*map(Path,parent['old_history_files']),ROOT/'stage/runs/t405_wide_admission/episodes/1.npz',DOC/'T408_PLAN.md',DOC/'RECOVERY_PRIMARY_SOURCES.md',DOC/'T406_FINDINGS.md',DOC/'T407_AUDIT.json',DOC/'T407_ORIGINAL_RECEIPT.json',Path(src['source_protocol']),ROOT/'stage/runs/t404_last_narrow/protocol.json',ROOT/'stage/runs/t404_last_narrow/cases.jsonl',ROOT/'stage/runs/t404_last_narrow/audit.json',ROOT/'stage/runs/t405_wide_admission/audit.json',ROOT/'stage/runs/t405_wide_admission/summary.json',ROOT/'stage/runs/t406_wide_full_route/audit.json',ROOT/'stage/runs/t406_wide_full_route/summary.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T404.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T405.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T406.json',ROOT/'stage/experiments/t406_wide_full_route/t406_core.py',ROOT/'stage/experiments/t404_last_narrow/t404_core.py',ROOT/'stage/experiments/t398_warm_recovery_rl/t398_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t348_causal_sequence_warmstart/t348_core.py',ROOT/'stage/runs/t393_safe_demo_pg/candidate.pt.gz',ROOT/'stage/runs/t402r_offline_night/candidate.pt.gz']
    for file in HERE.glob('*.py'):ast.parse(file.read_text())
    p={k:src[k] for k in ['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','deployment']}
    p.update(id='T408_JOINT_WIDE_LEFT_RIGHT_SINGLE_ACTOR',inputs={str(f):sha(f) for f in sorted(set(paths))},authorization='2026-11-07 user continue and reach global; narrow paused, separately bounded joint wide pilot before6->7->8',initial_model=str(PARENT),initial_sha256=sha(PARENT),initial_policy_hash='3a68dca0659f909cac7f334ba230712ee39461957758fcee86e273caf596de58',old8_jobs=old8,centres=centres,main_jobs=jobs,fresh_jobs=fresh,reload_jobs=reload,left5_jobs=left,right_job=right,left_history_files=src['wide_history_files'],right_history_file=str(ROOT/'stage/runs/t405_wide_admission/episodes/1.npz'),preflight_old_history_files=parent['old_history_files'],expected_old8_array_digests={j['id']:next(r['array_digest'] for r in oldrows if r['role']=='final_old8_'+j['id']) for j in old8},rl_jobs=[jobs[0],jobs[27]],seed=20261107408,consolidation_steps=200,old_weight=1.,broad_old_weight=1.,left_weight=.5,right_weight=.5,final_bytes=15*2**20,full_bytes=400*1024,max_commands=800,max_physics=18000000,main_seconds=7200,learning_seconds=3600,audit_seconds=1200,backup_seconds=600,source_seconds=120,all_seconds=9120,narrow_paused=True,task_version='wide117m_80s_shift0p6',joint_gate='all samecandidate old8+LEFT5+RIGHTcentre 14 safeGOAL, no successIDswap; failure wholeActorAdamrollback and stop consolidation mechanism',PPO='only if joint accepted, freshAdam then2 fixed rounds LEFT_00->RIGHT_00 each4new innovations+4exact,8live8shadow; numeric+.5 and joint14 beforekeep',full_gate='samefinalwide54+old275allsafe before16newwide then78namedcandidate native payload+integration exact; audit/remote full raw before6mstable; no7/8codeorlaunch',scientific_scope='joint factual support consolidation/PPO engineering pilot, no isolatedcausality/originality/safetycertificate; trainingonlymulti-source no runtime routing',stop='no grids/restarts/extension; any shared evidence/numeric contract failure stops; jointreject keepsT404 and nofullsweep; max2PPOretentionattempts',sim2real='single120true recurrentActor, samewheel5force120, fullvehiclecollision; actualsensor/actuator calibration and independentphysicalvalidation pending')
    # Reserve every registered branch, including conditional full acceptance.
    caps=lambda seq:sum(j['case']['step_cap'] for j in seq)
    worst=caps(old8)+caps(left)+caps([right])+4*(caps(old8)+caps(left)+caps([right]))+16*40000+caps(centres)*2+caps(jobs)+caps(base['final_old_jobs'])+caps(fresh)+caps(reload)
    commands=4+14+4*14+16+4+54+275+16+78
    own=21*p['full_bytes']+6*224*1024+650*5*1024+int(1.5*2**20)
    used=N.K.S.used(ROOT/'stage/runs');assert own<=p['final_bytes'] and worst<=p['max_physics'] and commands+100<=p['max_commands'];assert used+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert p['raw_sigma']==parent['raw_sigma']==[.2,.2] and p['lr']==parent['lr'] and p['left_weight']==p['right_weight']
    ledger={};ids={j['id'] for j in fresh};consumed=[]
    for file in sorted((ROOT/'stage/runs').rglob('*.jsonl')):
        if file.name not in ['cases.jsonl','replays.jsonl','evaluations.jsonl']:continue
        ledger[str(file)]=sha(file)
        for line in file.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(r,dict) and any(isinstance(v,str) and any(v==i or v.endswith('_'+i) for i in ids) for v in [r.get(k) for k in ['id','job_id','role','task_id']]):consumed.append(str(file))
    assert not consumed
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'fresh_admission.json',dict(state='PASS_WIDE_HOLDOUT_UNCONSUMED',job_ids=sorted(ids),locked_ledger_SHA=ledger,consumed=[]))
    save(DOC/'T408_BUDGET.json',dict(state='FULL_WORST_CASE_RESERVED',global_before=used,global_full_worst=used+p['final_bytes'],global_limit=2**30,own=p['final_bytes'],worst_bytes=own,worst_physics=worst,main_worst_commands=commands,preflight_zero_physical_commands_reserved=100,max_commands=p['max_commands'],full_slots=21,model_slots=6,compact_slots=650,metadata=int(1.5*2**20),max_live_Adam=216,max_shadow_Adam=216,historical_audit_Adam=216,preflight_discarded_Adam=8))
    print('T408_REGISTERED_NOT_STARTED',worst,own,flush=True)
def preflight():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name in ['run.py','audit.py','night.py']:
        spec=importlib.util.spec_from_file_location('stage408_actual_'+Path(name).stem,HERE/name);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def check(c):
            bad={v.argval for v in dis.get_instructions(c) if v.opname=='LOAD_GLOBAL' and v.argval not in m.__dict__ and not hasattr(builtins,v.argval)}
            for v in c.co_consts:
                if isinstance(v,types.CodeType):bad|=check(v)
            return bad
        for fn in vars(m).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==m.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    q,opt=learner(base,p);assert q.phash()==p['initial_policy_hash'];broad=N.old_data(q,base,lambda:None);assert len(broad)==211 and {g['task'] for g in broad}==set(range(8))
    old=[N.S.arrays(f) for f in p['preflight_old_history_files']];g=groups(q,old,p,broad);right=N.S.arrays(p['right_history_file']);STAGE404_RUNTIME.audit_array(q,right,p)
    with torch.no_grad():
        for z in g['old']:z['target']=q.sequence(z['x'],z['b']).detach().clone()
    before=L.state(q,opt);shadow,so=learner(base,p);L.load(shadow,so,before);calls={'live':0,'shadow':0}
    def event(which):
        def emit(where,k):
            if where=='AFTER':calls[which]+=1
            with (OUT/'preflight_optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,where,k])+'\n')
        return emit
    short=dict(p,consolidation_steps=2,epochs=2);u=consolidate(q,opt,g,short,lambda:None,event('live'));v=consolidate(shadow,so,g,short,lambda:None,event('shadow'));assert u==v and L.statehash(q,opt)==L.statehash(shadow,so)
    mature=L.state(q,opt);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before);L.load(q,opt,mature);L.load(q,opt,before)
    a=copy.deepcopy(right);x=torch.tensor(a['states']);q.reset_noise();q.sample=True;q.rng=np.random.default_rng(408);h=None
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    a.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});synthetic=[]
    for k in range(4):z=copy.deepcopy(a);z['rewards'][-1]+=.1*k;synthetic.append(z)
    L.load(shadow,so,before);u=rl_fit(q,opt,synthetic,g,short,0,lambda:None,event('live'));v=rl_fit(shadow,so,synthetic,g,short,0,lambda:None,event('shadow'));assert u==v and L.statehash(q,opt)==L.statehash(shadow,so);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before);L.export(q,opt,OUT/'preflight_clone.pt.gz');assert calls==dict(live=4,shadow=4)
    rng=np.random.default_rng(408);padded={}
    for k,v in N.S.packed(a).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape;padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    padded.update(ownership=rng.integers(0,2,400,np.uint8),hold_sha256=rng.integers(0,256,(400,32),np.uint8),task_goal_xy=np.zeros(2),scene_sha256=np.zeros(32,np.uint8));assert len(N.R.blob(padded))<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);orig=copy.deepcopy(m.ACTIVE);geom=[]
    try:
        for j in [*p['main_jobs'],*p['fresh_jobs'],*p['left5_jobs']]:
            m.ACTIVE=copy.deepcopy(orig);m.ACTIVE['circle_shift_x_m']=j['circle_shift_x_m'];env,obs,_=N.K.R.task_env(create,j)
            try:obs=N.K.C.perturb_anchor(env,obs,j);assert not env._collision();geom.append(dict(id=j['id'],scene=N.K.C.model_hash(env),native0=digest(N.K.R.integration(env))))
            finally:w.events.pop(id(env),None);env.close()
    finally:w.close()
    counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        vals=pool.request([dict(model=str(OUT/'preflight_clone.pt.gz'),job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'geometry.json',geom);save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=8,zero_physics_commands=len(geom)+1+4+4,actual_recursive_main_audit_driver_globals=True,actual_writer_lock=True,actual_T404_RIGHT_history_density_hidden_reward_exact=True,old211_all8task_support=True,equal_side_weight=True,all4worker_READY_cancel_exit=True,full_incompressible_bytes=len(N.R.blob(padded)),warm_and_PPO_live_shadow_Adam_rollback=True,synthetic_not_navigation=True));N.manifest(OUT);print('T408_PREFLIGHT_PASS_ZERO_NATIVE_8_DISCARDED_ADAM',flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/(sys.argv[1]+'_error.json'),dict(error=traceback.format_exc(),native='see zero-native preflight or status',restart=False))
        raise
