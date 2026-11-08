"""Unique once-only frozen evaluation, never resumes an interrupted trainer."""
from t424_eval import *
import subprocess,signal,importlib.util,dis,builtins,types
ACTIVE=None
def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True);prior=ROOT/'stage/runs/t422_feedback_recovery_night';old=read(prior/'protocol.json')
    keys=['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','left5_jobs','right_job','left_history_files','right_history_file','main_jobs','fresh_jobs','reload_jobs','centres','deployment','initial_model','full_bytes','query_full_bytes','fresh_ledger','inputs','phase']
    p={key:copy.deepcopy(old[key]) for key in keys}
    model=prior/'learning/blocks/3/before.pt.gz';assert sha(model)=='73490bd3d6d7c73c9a351f30979010f90ef48be012c9508403068ae9d05dc38f'
    assert model.read_bytes()==(prior/'learning/blocks/2/rl_before.pt.gz').read_bytes()
    previous=[json.loads(line) for line in (prior/'learning/cases.jsonl').read_text().splitlines()]
    history=[row for row in previous if row['role'] in ['BC2_proposal_old8','BC2_proposal_left5','BC2_proposal_right']];assert len(history)==14 and all(x['success'] for x in history)
    base=E.N.science(p);alljobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job'],*p['main_jobs'],*base['final_old_jobs'],*p['fresh_jobs'],*p['reload_jobs']]
    worst=sum(j['case']['step_cap'] for j in alljobs);assert worst==10926739 and len(alljobs)==437
    extra=[]
    for folder in sorted((ROOT/'stage/runs').iterdir()):
        prefix=folder.name.split('_',1)[0]
        if prefix.startswith('t') and prefix[1:].isdigit() and 412<=int(prefix[1:])<=423:
            extra += [str(f) for f in folder.rglob('*.jsonl') if f.name in ['cases.jsonl','replays.jsonl']]
    fresh_ids={j['id'] for j in p['fresh_jobs']}
    for f in extra:
        for line in Path(f).read_text().splitlines():
            row=json.loads(line);assert row.get('job_id') not in fresh_ids,('Fresh already consumed',f,row.get('job_id'))
    p['fresh_ledger'].update({f:sha(f) for f in extra})
    p.update(id='T424_FROZEN_SERIALIZED_SIX_METRE_ACCEPTANCE',out=str(OUT),student_start_model=str(model),student_start_sha256=sha(model),expected_policy_hash='6ad474ec960d9f8c61a62abefe958b7b8e9bb057a2a23631916c53db06c31051',student_policy_hash='6ad474ec960d9f8c61a62abefe958b7b8e9bb057a2a23631916c53db06c31051',
        precheck_history_rows=history,old275_jobs=base['final_old_jobs'],main_seconds=2400,audit_seconds=600,backup_seconds=300,source_seconds=180,all_seconds=3600,max_physics=11000000,max_main_commands=441,preflight_zero_commands=8,max_commands=449,final_bytes=48*2**20,
        Actor_updates=0,optimizer_steps=0,training_provenance='T422 optimizer historical reconstruction failed first AFTER0; unresolved, original FAIL preserved. New frozen navigation evaluation is independent and does not certify old training.',
        scientific_scope='Single frozen serialized Actor; 14 prior native histories exact, full old275+wide54; only both allsafe permit16 heldout, then78 independent complete native model reloads. No policy selection or parameter/seed/threshold adjustment.',
        no_runtime_teacher_or_planner=True,no_parameter_or_seed_grid=True,no_seven_eight=True,narrow_paused=True,safe_T255_frozen=True)
    inputs=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{str(model),str(prior/'learning/blocks/2/bc_proposed.pt.gz'),str(prior/'learning/blocks/2/result.json'),str(prior/'learning/cases.jsonl'),str(ROOT/'stage/docs/overnight_20261107/T422_PARTIAL_OPTIMIZER_DIAGNOSTIC.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T422-INTERRUPTED.json')}
    p['inputs']={f:sha(f) for f in sorted(inputs)}
    used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics_worst=worst,physics_cap=11000000,main_commands_worst=441,all_commands_worst=449,full_payload_max=92,full_each_bytes=p['full_bytes'],compact_max=275,compact_each_bytes=5120,own_bytes=p['final_bytes'],global_before=used,global_full_reserved=used+p['final_bytes'],main_seconds=2400,audit_seconds=600,backup_seconds=300,source_seconds=180,all_seconds=3600,Actor_updates=0,optimizer_steps=0,incremental_fresh_ledger_files=len(extra),inherited_training_audit_not_overridden=True))
    print('T424_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=frozen(p);full=torch.load(E.io.BytesIO(E.gzip.decompress(Path(p['student_start_model']).read_bytes())),weights_only=False,map_location='cpu');assert full['policy_hash']==p['expected_policy_hash'] and full['hashes']['learned']==digest(full['state']['learned'])
    bc=torch.load(E.io.BytesIO(E.gzip.decompress((ROOT/'stage/runs/t422_feedback_recovery_night/learning/blocks/2/bc_proposed.pt.gz').read_bytes())),weights_only=False,map_location='cpu');assert bc['hashes']['learned']==full['hashes']['learned']
    source=read(ROOT/'stage/runs/t422_feedback_recovery_night/learning/initial_arrays.json')[0];a=E.N.S.arrays(source['path']);x=torch.tensor(a['states']);b=q.base_actions(x)
    with torch.no_grad():
        expected=q.sequence(x,b);h=None;actual=[]
        for row in x:
            _,loc,h,_=q.step(row[None],h);actual.append(loc)
        maximum=float((torch.cat(actual)-expected).abs().max());assert maximum<2e-12
    for name in ['t424_eval.py','runner.py','audit.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage424_entry_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    pref=OUT/'preflight_native';pref.mkdir();counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(pref),native_stop_epoch=time.time()+120),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['main_jobs'][0],seed=None,cancel_probe=True)]*4);assert all(x['state']=='CANCEL_PASS' for x in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,Actor_updates=0,optimizer_steps=0,zero_physics_commands=8,actual_four_workers_cancel_before_native_and_exit=True,same_accepted_BC2_actor_SHA=True,complete_serialized_model_SHA=True,carried_sequence_step_max_error=maximum,recursive_actual_entry_globals=True,no_runtime_teacher_loaded=True,source_hashes=True,old_optimizer_history_FAIL_remains=True));E.N.manifest(OUT);print('T424_PREFLIGHT_PASS_0NATIVE_0ADAM',flush=True)
def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time()
    with (DOC/'CONSOLE.log').open('xb') as f:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T424','--why=Bounded frozen six metre acceptance','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+p['all_seconds'],source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),Actor_updates=0,optimizer_steps=0,no_old_job_restart=True));print('T424_ACTUALLY_LAUNCHED',child.pid,flush=True)
def stop_active():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait(timeout=3)
def call(file,args,seconds,deadline,log):
    global ACTIVE
    with Path(log).open('xb') as f:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:assert ACTIVE.wait(timeout=min(seconds,deadline-time.time()-5))==0,file
        except BaseException:stop_active();raise
        finally:ACTIVE=None
def driver():
    p=read(OUT/'protocol.json');verify(p);launch=read(OUT/'launch.json');start=time.time();deadline=launch['deadline'];state='FAILED';error=None;remote='PENDING'
    assert not (OUT/'driver_started.json').exists();save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def interrupted(signum,frame):raise TimeoutError('T424 hard stop '+str(signum))
    for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(deadline-time.time()-3)))
    try:
        call('t424_eval.py',[],p['main_seconds'],deadline-600,DOC/'MAIN.log')
        assert not Path('/proc',str(read(OUT/'main_started.json')['pid'])).exists()
        call('audit.py',[],p['audit_seconds'],deadline-300,DOC/'AUDIT.log');assert read(OUT/'audit.json')['state']=='PASS_FROZEN_ACCEPTANCE_EVIDENCE_ONLY';state='COMPLETE_AUDITED_REMOTE_PENDING'
        try:
            call('transport.py',['T424'],p['backup_seconds'],deadline-180,DOC/'BACKUP.log');remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T424.json')['state'];state='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
    except BaseException:error=traceback.format_exc();stop_active();save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
    finally:
        stop_active();signal.alarm(0);save(DOC/'FINAL_REPORT.json',dict(state=state,finished_epoch=time.time(),elapsed_s=time.time()-start,deadline=deadline,error=error,remote=remote,new_Actor_updates=0,new_optimizer_steps=0,old_training_audit='FAIL_RECONSTRUCTION_CAUSE_UNRESOLVED',source_sync='Root outcome commit and exact synchronization remain separate.'));print('T424_FINISHED',state,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
