"""Register and finalize a single bounded handoff test, never train an Actor."""
from science import *
import subprocess,signal,importlib.util,types,dis,builtins,ast
def prepare():
    assert not OUT.exists();origin=ROOT/'stage/runs/t418_evening_repair_chain';p=copy.deepcopy(read(origin/'repair/protocol.json'));draft=read(DOC/'NEXT_AFTER_T419.json');find=read(DOC/'T419_FINAL_REPORT.json')
    assert find['state']=='COMPLETE_AUDITED_FULL_REMOTE_READBACK_VERIFIED' and find['geometry'][1]['proposed_handoff_hold']==17
    model=ROOT/draft['student_source'];assert sha(model)==draft['student_model_sha256']
    rows=[json.loads(v) for v in (origin/'repair/cases.jsonl').read_text().splitlines()];parent=next(r for r in rows if r['role']=='initial_left5_wide_dev_3');oldquery=next(r for r in rows if r['role']=='teacher_0_left_carry_wide_dev_3')
    job=next(j for j in p['left5_jobs'] if j['id']=='wide_dev_3');assert job['case']['step_cap']==40000
    p.update(id='T420_SINGLE_FIXED_LEFT_HOLD17_RECOVERY',out=str(OUT),phase='repair',student_start_model=str(model),student_start_sha256=sha(model),student_policy_hash=draft['student_policy_hash'],teacher_first=17,job=job,
        case=dict(id='fixed_left_handoff17',job=job,first=17,teacher_kind='left_carry',source_control=False),parent_history=parent,old_failed_query=oldquery,
        final_bytes=3*2**20,max_physics=80000,max_commands=14,main_seconds=240,audit_seconds=120,backup_seconds=180,source_seconds=120,all_seconds=720,
        native_stop_absolute=1794055500,all_stop_absolute=1794056400,learning_stop_absolute=1794055500,
        time_authorization='User explicitly started next step at20:25 on2026-11-07; new independent fixed handoff only, earlier T41820:20native deadline unchanged; all21:00 retained.',
        max_navigation_commands=6,preflight_zero_physics_commands=8,preflight_Adam=0,live_Adam=0,Actor_updates=0,exact_replay_even_if_navigation_failure=True,no_automatic_training=True)
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(model),str(DOC/'NEXT_AFTER_T419.json'),str(DOC/'T419_FINAL_REPORT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T419.json'),parent['path'],oldquery['path']}
    p['inputs']={f:sha(f) for f in sorted(files)}
    assert time.time()+p['all_seconds']<p['all_stop_absolute'] and time.time()+p['main_seconds']+180<p['native_stop_absolute']
    assert E.N.K.S.used(ROOT/'stage/runs')+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T420_BUDGET.json',dict(state='SINGLE_NEW_HANDOFF_NOT_PRIOR_BUDGET_EXTENSION',own_bytes=p['final_bytes'],physics=80000,main_commands=6,preflight_commands=8,all_commands=14,Actor_updates=0,Adam=0,main_seconds=240,audit_seconds=120,backup_seconds=180,source_seconds=120,all_seconds=720,absolute_all=1794056400,global_reserved=E.N.K.S.used(ROOT/'stage/runs')+p['final_bytes']))
    print('T420_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();ban_optimization();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Existing mixed control contract is checked read-only at its real20hold
    # handoff. It is not counted as new hold17 recovery or new navigation.
    q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);assert q.phash()==p['student_policy_hash'];a=E.N.S.arrays(p['old_failed_query']['path']);oldcase=dict(p['case'],first=20);E.audit_query(q,a,oldcase,p)
    for name in ['science.py','runner.py']:
        spec=importlib.util.spec_from_file_location('stage420_preflight_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    prefout=OUT/'preflight_native';prefout.mkdir();E.set_out(prefout);counts=mp.get_context('spawn').RawArray('q',5);pool=E.Pool(dict(p,out=str(prefout),native_stop_epoch=time.time()+120),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];request=dict(model=p['student_start_model'],job=p['job'],first=17,seed=None,query=True,case=p['case'],cancel_probe=True)
        results=pool.request([request]*4);assert all(r['state']=='CANCEL_PASS' for r in results) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,zero_commands=8,all_source_SHA=True,student_model_exact=True,actual_teacher_mixed_history_contract_readonly=True,actual_four_workers_hold17_request_zero_step_cancel_and_exit=True,workers=ids,recursive_entry_globals=True,no_new_recovery_claim=True));E.N.manifest(OUT)
    print('T420_PREFLIGHT_PASS_0NATIVE_0ADAM',flush=True)
def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    now=time.time();assert now+p['all_seconds']<p['all_stop_absolute'] and now+p['main_seconds']<p['native_stop_absolute']
    with (DOC/'T420_CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T420','--why=Single fixed recovery test','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=min(now+p['all_seconds'],p['all_stop_absolute']),source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),Actor_updates=0,optimizer_steps=0));print('T420_ACTUALLY_LAUNCHED',child.pid,flush=True)
def driver():
    p=read(OUT/'protocol.json');verify(p);launch=read(OUT/'launch.json');deadline=launch['deadline'];start=time.time();active=None;state='FAILED'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def stop():
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
    def interrupted(*a):raise TimeoutError('T420 absolute stop')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(deadline-time.time()-3)))
    try:
        for file,arg,seconds in [('science.py','main',p['main_seconds']),('science.py','audit',p['audit_seconds']),('transport.py','T420',p['backup_seconds'])]:
            with (DOC/('T420_'+arg+'.log')).open('xb') as log:
                active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),arg],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                assert active.wait(timeout=min(seconds+2,deadline-time.time()-5))==0,(file,arg);active=None
        assert read(OUT/'audit.json')['state']=='PASS';assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T420.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';state='COMPLETE_AUDITED_REMOTE_VERIFIED'
    except BaseException:stop();save(DOC/'T420_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_restart=True))
    finally:
        stop();signal.alarm(0);save(DOC/'T420_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),finished_epoch=time.time(),elapsed_s=time.time()-start,deadline=deadline,Actor_updates=0,optimizer_steps=0,no_automatic_training=True));print('T420_DRIVER_FINISHED',state,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
