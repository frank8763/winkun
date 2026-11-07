"""Registration and real-path zero-native admission for a separately versioned task."""
from t406_core import *
import ast,importlib.util,types,dis,builtins
def version(j):
    j=copy.deepcopy(j)
    if j['id'].startswith('new070_80_'):j['id']='wide117_80_'+j['id'][10:]
    else:j['id']='wide117_80_'+j['id']
    j['case']['step_cap']=40000;j.update(wide_course=True,circle_shift_x_m=.6,expected_initial120_sha256=None);return j
def prepare():
    assert not OUT.exists();last=read(ROOT/'stage/runs/t405_wide_admission/summary.json');assert last['state']=='COMPLETE' and read(ROOT/'stage/runs/t405_wide_admission/audit.json')['state']=='PASS' and not Path('/proc',str(last['pid'])).exists()
    src=read(ROOT/'stage/runs/t393_safe_demo_pg/protocol.json');legacy=read(ROOT/'stage/runs/t366_factual_history/protocol.json');jobs=[version(j) for j in src['main_jobs']];fresh=[version(j) for j in src['fresh_jobs']]
    reload=[version(j) if j['id'].startswith('new070_80_') else copy.deepcopy(j) for j in src['reload_jobs']]
    widejobs=[]
    for item in legacy['selected'][:5]:
        j=copy.deepcopy(item['job']);j.update(circle_shift_x_m=.6,expected_initial120_sha256=None);widejobs.append(j)
    histories=[ROOT/'stage/runs/t366_factual_history/episodes'/f'wide_dev_{k}.npz' for k in range(5)]
    paths=[*HERE.glob('*.py'),LEGACY,*histories,ROOT/'stage/runs/t366_factual_history/protocol.json',ROOT/'stage/runs/t366_factual_history/audit.json',ROOT/'stage/runs/t366_factual_history/cases.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T366.json',ROOT/'stage/runs/t353_temporal_exploration/summary.json',ROOT/'stage/runs/t405_wide_admission/summary.json',ROOT/'stage/runs/t405_wide_admission/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T405.json',DOC/'T406_PLAN.md',DOC/'T404_T405_FINDINGS.md',ROOT/'stage/experiments/t404_last_narrow/t404_core.py',ROOT/'stage/experiments/t398_warm_recovery_rl/t398_learning.py',ROOT/'stage/experiments/t392_demo_closed_loop/t392_learning.py',ROOT/'stage/experiments/t393_safe_demo_pg/t393_learning.py',ROOT/'stage/experiments/t348_causal_sequence_warmstart/t348_core.py',Path(src['source_protocol'])]
    paths.extend([ROOT/'stage/runs/t393_safe_demo_pg/candidate.pt.gz',ROOT/'stage/runs/t402r_offline_night/candidate.pt.gz',DOC/'T406_IMPORT_FAILED_001.json',DOC/'T406_REGISTRATION_FAILED_001.json'])
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p={k:src[k] for k in ['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','deployment']}
    p.update(id='T406_WIDE117_80_FULL_ROUTE_PPO',inputs={str(f):sha(f) for f in sorted(set(paths))},authorization='2026-11-07 user starts next experiment; narrow paused after final failure; advance separately registered wide6 then7/8 only on complete gate',main_jobs=jobs,fresh_jobs=fresh,reload_jobs=reload,wide5_jobs=widejobs,wide_history_files=[str(f) for f in histories],legacy_model=str(LEGACY),legacy_model_sha256=sha(LEGACY),legacy_policy_hash='da6212f7f0ac915f2134f98ca3b8e3c74cb1d1f9ff2e6cca05b0e3db051a93b8',initialization='load legacy mean/scale/encoder/core/head; explicitly zero added adapter final weight+bias; exact5history function proof; freshAdam6864, frozen future_head; do not merge stale adapter or legacy Adam',seed=20261107406,rounds=2,old_weight=1.,wide_weight=1.,final_bytes=int(12.5*1024**2),full_bytes=400*1024,max_commands=1100,max_physics=20000000,main_seconds=7200,learning_seconds=5400,audit_seconds=1200,backup_seconds=600,all_seconds=9120,scientific_scope='wide task re-entry and joint old-repair/route-PPO pilot, not novelalgorithm or isolated effect; initial fullold275 measured beforetrain',target_selection='initial failed wide54 in canonical order, first2 across2rounds; if none then initialold275 failures; 4same-task independent innovations perbatch; no adaptive seed/taskgrid',acceptance='numeric maxlogratio<=.5 and all retained old20 success IDs and historical wide5 success IDs remain; rejects whole Actor+Adam; two rejections stop',promotion='same final wide54 and originalold275 all safe then16newwide holdout and78serialized native payload/integration exact; no7/8 automatic',task_version='wide117m_80s_shift0p6',narrow_paused=True,sim2real='samefullvehicle/tyres/collision/wheel5/force120 and causal120 singleActor; no runtime maps/planner/waypoint/teacher/router; real sensor/actuator interface not yet calibrated')
    base=read(p['source_protocol']);assert len(base['old_jobs'])==20 and len(jobs)==54 and len(fresh)==16 and len(reload)==78
    worstphysics=2*(sum(j['case']['step_cap'] for j in [*jobs,*base['final_old_jobs']])+5*40000)+16*40000+2*sum(j['case']['step_cap'] for j in [*base['old_jobs'],*widejobs])+sum(j['case']['step_cap'] for j in [*fresh,*reload])
    worstbytes=13*p['full_bytes']+6*224*1024+900*5*1024+int(1.5*1024**2)
    used=N.K.S.used(ROOT/'stage/runs');assert worstphysics<=p['max_physics'] and worstbytes<=p['final_bytes'] and used+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    ledger={};ids={j['id'] for j in fresh};consumed=[]
    for f in sorted((ROOT/'stage/runs').rglob('*.jsonl')):
        if f.name not in ['cases.jsonl','replays.jsonl','evaluations.jsonl']:continue
        ledger[str(f)]=sha(f)
        for line in f.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(r,dict) and any(isinstance(v,str) and any(v==j or v.endswith('_'+j) for j in ids) for v in [r.get(k) for k in ['id','job_id','role','task_id']]):consumed.append(str(f))
    assert not consumed
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'fresh_admission.json',dict(state='PASS_NO_PRIOR_WIDE_VERSION_NAVIGATION',job_ids=sorted(ids),locked_ledger_SHA=ledger,consumed=[]))
    save(DOC/'T406_BUDGET.json',dict(state='FULL_WORST_CASE_RESERVED',global_before=used,global_full_worst=used+p['final_bytes'],global_limit=2**30,own=p['final_bytes'],worst_bytes=worstbytes,worst_physics=worstphysics,max_commands=1100,full_slots=13,models=6,compact_slots=900,metadata_bytes=int(1.5*1024**2),max_live_Adam=16,max_shadow_Adam=16,max_historical_audit_Adam=16,preflight_discarded_Adam=4));print('T406_REGISTERED_NOT_STARTED',worstphysics,worstbytes,flush=True)
def preflight():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage406_actual_'+Path(name).stem,HERE/name);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def check(c):
            bad={v.argval for v in dis.get_instructions(c) if v.opname=='LOAD_GLOBAL' and v.argval not in m.__dict__ and not hasattr(builtins,v.argval)}
            for v in c.co_consts:
                if isinstance(v,types.CodeType):bad|=check(v)
            return bad
        for fn in vars(m).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==m.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    q,opt=learner(base,p);legacy=N.S.read_model(LEGACY);assert digest({k:q.learned()[k] for k in legacy})==digest(legacy);proof=[]
    for file in p['wide_history_files']:
        a=N.S.arrays(file);q.reset_noise();q.sample=False;h=None;locs=[]
        with torch.no_grad():
            for i,x in enumerate(a['states']):
                hb=np.zeros(16) if h is None else h[0,0].numpy().copy();cmd,loc,h,_=q.step(torch.tensor(x[None]),h);assert equal(hb,a['hidden_before'][i]) and equal(cmd[0].numpy(),a['actions'][i]) and equal(loc[0].numpy(),a['locations'][i]);locs.append(loc[0])
            seq=q.sequence(torch.tensor(a['states']),q.base_actions(torch.tensor(a['states'])));err=float((seq-torch.stack(locs)).abs().max());assert err<1e-10 and equal(h[0,0].numpy(),a['last_hidden_after'])
        proof.append(dict(file=file,executed_actions_exact=True,serial_locations_hidden_exact=True,sequence_max_error=err))
    old,wide=supports(q,base,p,lambda:None);a=N.S.arrays(p['wide_history_files'][0]);x=torch.tensor(a['states']);q.reset_noise();q.sample=True;q.rng=np.random.default_rng(406);h=None
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    aa=copy.deepcopy(a);aa.update({k:np.asarray([v[k] for v in q.tape]) for k in q.tape[0]});synthetic=[]
    for i in range(4):v=copy.deepcopy(aa);v['rewards'][-1]+=.1*i;synthetic.append(v)
    before=L.state(q,opt);shadow,so=learner(base,p);L.load(shadow,so,before);short=dict(p,epochs=2);u=fit(q,opt,synthetic,short,old,wide,0,lambda:None,lambda *_:None);v=fit(shadow,so,synthetic,short,old,wide,0,lambda:None,lambda *_:None);assert u==v and L.statehash(q,opt)==L.statehash(shadow,so)
    mature=L.state(q,opt);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before);L.load(q,opt,mature);L.load(q,opt,before);L.export(q,opt,OUT/'preflight_clone.pt.gz')
    rng=np.random.default_rng(406);padded={}
    for k,v in N.S.packed(aa).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(x) else v.shape;padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    padded.update(ownership=rng.integers(0,2,400,np.uint8),hold_sha256=rng.integers(0,256,(400,32),np.uint8),task_goal_xy=np.zeros(2),scene_sha256=np.zeros(32,np.uint8));assert len(N.R.blob(padded))<=p['full_bytes']
    m,native,create,w,_=N.S.P.C.B.setup(OUT/'preflight_native',base);orig=copy.deepcopy(m.ACTIVE);geom=[]
    try:
        for j in [*p['main_jobs'],*p['fresh_jobs'],*p['wide5_jobs']]:
            m.ACTIVE=copy.deepcopy(orig);m.ACTIVE['circle_shift_x_m']=j['circle_shift_x_m'];env,obs,_=N.K.R.task_env(create,j)
            try:obs=N.K.C.perturb_anchor(env,obs,j);assert not env._collision();geom.append(dict(id=j['id'],scene=N.K.C.model_hash(env),native0=digest(N.K.R.integration(env))))
            finally:w.events.pop(id(env),None);env.close()
    finally:w.close()
    counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        vals=pool.request([dict(model=str(OUT/'preflight_clone.pt.gz'),job=p['wide5_jobs'][0],first=0,seed=None,cancel_probe=True)]*4);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'legacy_function_proof.json',dict(state='PASS_EXACT_5_FACTUAL_HISTORY_FUNCTION',legacy_sha256=sha(LEGACY),legacy_policy_hash=p['legacy_policy_hash'],new_sampling_policy_hash=q.phash(),fresh_Adam=True,zero_adapter_output=True,records=proof));save(OUT/'geometry.json',geom)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=4,prior_discarded_Adam=p.get('prior_preflight_discarded_Adam',0),total_preflight_discarded_Adam=4+p.get('prior_preflight_discarded_Adam',0),zero_physics_commands=len(geom)+1+4+4,actual_main_audit_driver_recursive_globals=True,actual_writer_lock=True,legacy5function_exact=True,all4worker_READY_cancel_exit=True,incompressible_full_bytes=len(N.R.blob(padded)),live_shadow_and_mature_Adam_rollback=True));N.manifest(OUT);print('T406_PREFLIGHT_PASS_ZERO_NATIVE_4_DISCARDED_ADAM',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
