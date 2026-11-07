"""Register one new likelihood/corrective-data pilot within today's remainder."""
from t412_learning import *
import ast,dis,types,builtins,importlib.util,fcntl,subprocess
def prepare():
    assert not OUT.exists();prior=ROOT/'stage/runs/t411_closed_teacher_recovery';s=read(prior/'summary.json');assert s['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS' and not Path('/proc',str(s['pid'])).exists();assert read(DOC/'T411_FINAL_REPORT.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    previous=read(ROOT/'stage/runs/t408_joint_wide/protocol.json');p={k:previous[k] for k in ['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','left5_jobs','right_job','left_history_files','right_history_file','main_jobs','fresh_jobs','reload_jobs','centres','deployment']}
    student=ROOT/'stage/runs/t410_day_offline_chain/lateral/candidate.pt.gz';teacher=ROOT/'stage/runs/t406_wide_full_route/candidate.pt.gz';safe_history=ROOT/'stage/runs/t288_five_metre_presentation/right_full120.npz';queries=[prior/'episodes/recover4s_0.npz',prior/'episodes/recover4s_2.npz'];files=[*HERE.glob('*.py'),DOC/'T411_FINDINGS_AND_T412_PLAN.md',student,teacher,safe_history,*queries,prior/'protocol.json',prior/'summary.json',prior/'audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T411.json',ROOT/'stage/experiments/t410_day_offline_chain/day_core.py',Path(p['source_protocol'])]
    files.extend(Path(f) for f in previous['inputs'] if str(f).endswith('.py') or 'candidate.pt.gz' in str(f))
    for file in HERE.glob('*.py'):ast.parse(file.read_text())
    freshids={j['id'] for j in p['fresh_jobs']};ledger={}
    for f in sorted((ROOT/'stage/runs').rglob('*.jsonl')):
        if f.name not in ['cases.jsonl','replays.jsonl','evaluations.jsonl']:continue
        ledger[str(f)]=sha(f)
        for line in f.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(r,dict):assert not any(isinstance(v,str) and any(v==i or v.endswith('_'+i) for i in freshids) for v in [r.get(k) for k in ['id','job_id','role','task_id']]),f
    day=read(ROOT/'stage/runs/t410_day_offline_chain/day_ledger.json');actual=copy.deepcopy(day['actual']);actual['physics']+=s['physics'];actual['commands']+=s['commands']+read(prior/'preflight.json')['zero_physics_commands'];actual['main_wall_s']+=s['elapsed_s']
    p.update(id='T412_SAFE_FUNCTION_CLOSED_EXPERT_LIKELIHOOD',initial_model=str(student),safe_history_file=str(safe_history),preflight_query_files=[str(f) for f in queries],teacher_model=str(teacher),teacher_policy_hash='631103faeff6d840b94c43e6b10fb2464e537e38edd84f8c8d02f0784710278d',old_weight=1.,broad_old_weight=1.,left_weight=.25,right_weight=.25,query_weight=1.,bc_steps=400,teacher_first=20,query_pairs=[[0,1],[2,3]],seed=20261107412,full_bytes=400*1024,query_full_bytes=544*1024,final_bytes=20*2**20,max_physics=16000000,max_commands=900,main_seconds=5400,learning_seconds=3600,audit_seconds=900,backup_seconds=300,source_seconds=120,all_seconds=7200,learning_stop_absolute=1794048300.,native_stop_absolute=1794051000.,all_stop_absolute=1794052800.,minimum_cycle_seconds=600,fresh_ledger=ledger,prior_day_actual=actual,day_limits=day['limits'],inputs={str(f):sha(f) for f in sorted(set(files))},initialization='zero head andoldadapterfinal andextraoutput; freezealloriginal; 4450extraonly; executedcommands equalT255',old_support_loss='same-sigma latent Gaussian KL all8currentactualhistories average eachstep plusbalanced211; zero atsafeinitial, no trajectorycertificate',new_label_loss='existing float32boundarytail/interiorGaussian NLL onfullcurrentexpertqueries includingstudentprefix plusactualLEFT/RIGHTsupport',acceptance='old8/RIGHT/allretainedLEFTsuccessIDs nonregression, BCrollbackpreBC ActorAdam; acceptedBC freshAdam then4new/4exact/8PPO maxratio.5, RLrollbackpostBCfreshAdam',scientific_scope='parent/data/likelihood jointengineeringpilot notisolatedcausality ororiginality; teacherquery successnotstudentautonomy',narrow_paused=True,no_seven_eight=True)
    cap=lambda js:sum(j['case']['step_cap'] for j in js);base=read(p['source_protocol']);joint=cap([*p['old8_jobs'],*p['left5_jobs'],p['right_job']]);worst=6*joint+16*40000+8*40000+cap(p['centres'])*2+cap(p['main_jobs'])+cap(base['final_old_jobs'])+cap(p['fresh_jobs'])+cap(p['reload_jobs']);worstbytes=16*p['full_bytes']+8*p['query_full_bytes']+8*512*1024+550*5*1024+int(1.5*2**20)
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30;assert worst<p['max_physics'] and worstbytes<p['final_bytes'];assert actual['physics']+p['max_physics']<=day['limits']['physics'] and actual['commands']+p['max_commands']+8<=day['limits']['commands'] and actual['main_wall_s']+p['main_seconds']<=day['limits']['main_wall_s'];assert time.time()+p['all_seconds']<p['all_stop_absolute']
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'T412_BUDGET.json',dict(state='FULL_WORST_CASE_RESERVED',global_before=used,global_full_worst=used+p['final_bytes'],limit=2**30,own=p['final_bytes'],worst_bytes=worstbytes,worst_physics=worst,main_commands_worst=639,preflight_zero_physical_commands_reserved=8,max_commands=900,max_live_Adam=816,max_shadow_Adam=816,max_audit_historical_discarded_Adam=816,preflight_discarded_Adam=8,prior_day_actual=actual,day_limits=day['limits']));print('T412_REGISTERED_NOT_STARTED',worst,worstbytes,flush=True)

def preflight():
    N.offline();p=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage412_actual_'+Path(name).stem,HERE/name);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def check(c):
            bad={i.argval for i in dis.get_instructions(c) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(m) and not hasattr(builtins,i.argval)}
            for v in c.co_consts:
                if isinstance(v,types.CodeType):bad|=check(v)
            return bad
        for fn in vars(m).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==m.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
    set_out(OUT);base=N.science(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);assert sum(v.numel() for v in params(q))==4450 and not q.base.memory_on
    broad=N.old_data(q,base,lambda:None)
    with torch.no_grad():
        for g in broad:assert equal(q.sequence(g['x'],g['b']).clamp(-1,1).float(),g['b'].float())
    a=N.S.arrays(p['safe_history_file']);q.reset_noise();h=None
    with torch.no_grad():
        for x,u in zip(a['states'],a['actions']):v,_,h,_=q.step(torch.tensor(x[None]),h);assert equal(v[0].numpy(),u)
    oldfiles=read(ROOT/'stage/runs/t404_last_narrow/protocol.json')['old_history_files'];old=[N.S.arrays(f) for f in oldfiles];g=groups(q,old,p,broad);query=N.S.arrays(p['preflight_query_files'][0]);g['corrections']=[query_group(q,query)];before=L.state(q,opt);shadow,so=learner(base,p);L.load(shadow,so,before)
    # Prefix labels contribute: the existing teacher labels are used before
    # handoff too, independently of which policy physically owned the action.
    ref=g['corrections'][0];pred=q.sequence(ref['x'],ref['b']);prefixloss=-L.logp(pred[:20],ref['raw'][:20],ref['atom'][:20],p).mean();grad=torch.autograd.grad(prefixloss,params(q));assert sum(float(v.square().sum()) for v in grad)>0
    one=fit_bc(q,opt,g,p,lambda:None,lambda *_:None,steps=2);two=fit_bc(shadow,so,g,p,lambda:None,lambda *_:None,steps=2);assert one==two and L.statehash(q,opt)==L.statehash(shadow,so);mature=L.state(q,opt);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before);L.load(q,opt,mature);L.load(q,opt,before)
    q.reset_noise();q.rng=np.random.default_rng(412);q.sample=True;h=None;x=torch.tensor(a['states'])
    with torch.no_grad():
        for row in x:_,_,h,_=q.step(row[None],h)
    synthetic=copy.deepcopy(a);synthetic.update({k:np.asarray([r[k] for r in q.tape]) for k in q.tape[0]});items=[]
    for k in range(4):v=copy.deepcopy(synthetic);v['rewards'][-1]+=.1*k;items.append(v)
    L.load(shadow,so,before);short=dict(p,epochs=2);one=fit_ppo(q,opt,items,g,short,0,lambda:None,lambda *_:None);two=fit_ppo(shadow,so,items,g,short,0,lambda:None,lambda *_:None);assert one==two and L.statehash(q,opt)==L.statehash(shadow,so);L.load(q,opt,before);assert digest(L.state(q,opt))==digest(before);export(q,opt,OUT/'preflight_clone.pt.gz')
    rng=np.random.default_rng(412);padded={}
    for k,v in N.S.packed(query).items():
        shape=(401,*v.shape[1:]) if k.startswith('history_') else (400,*v.shape[1:]) if v.ndim and len(v)==len(query['states']) else v.shape;padded[k]=rng.normal(size=shape).astype(v.dtype) if np.issubdtype(v.dtype,np.floating) else rng.integers(0,100,size=shape,dtype=v.dtype)
    full=len(N.R.blob(padded));assert full<=p['query_full_bytes']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(OUT),native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        result=pool.request([dict(model=str(OUT/'preflight_clone.pt.gz'),job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4);assert all(x['state']=='CANCEL_PASS' for x in result) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=8,zero_physics_commands=8,safe_T255_commands_all211_and_native_RIGHThistory_exact=True,only4450extra_trainable=True,teacher_prefix_labels_nonzero_gradient=True,BC_and_PPO_live_shadow_mature_Adam_rollback=True,actual_four_worker_READY_cancel_exit=True,actual_writer_lock=True,actual_recursive_main_audit_driver_globals=True,incompressible_query_bytes=full,query_full_limit=p['query_full_bytes'],teacher_is_training_only=True));N.manifest(OUT);print('T412_PREFLIGHT_PASS_ZERO_NATIVE_8_DISCARDED_ADAM',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
