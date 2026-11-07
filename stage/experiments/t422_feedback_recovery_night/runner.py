"""Precommitted offline overnight chain, with real absolute stop deadlines."""
from t422_common import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,signal,importlib.util,dis,builtins,types
ACTIVE=None
def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True)
    previous=read(ROOT/'stage/runs/t418_evening_repair_chain/protect/protocol.json')
    keys=['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','left5_jobs','right_job','left_history_files','right_history_file','main_jobs','fresh_jobs','reload_jobs','centres','deployment','initial_model','safe_history_file','teacher_model','teacher_policy_hash','full_bytes','query_full_bytes','fresh_ledger','inputs','query_sources','initial_expected_policy_hash','initial_expected_success_ids','teacher_first']
    p={key:copy.deepcopy(previous[key]) for key in keys}
    old=read(ROOT/'stage/runs/t420_fixed_early_recovery/protocol.json');model=ROOT/'stage/runs/t417_representation_adaptation/blocks/0/bc_proposed.pt.gz'
    stop=datetime(2026,11,8,8,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    reference=ROOT/'stage/runs/t366_factual_history/episodes/wide_dev_3.npz'
    fixtures=[r for r in [json.loads(x) for x in (ROOT/'stage/runs/t418_evening_repair_chain/repair/cases.jsonl').read_text().splitlines()] if r['role'].startswith('initial_')]
    assert len(fixtures)==14
    p.update(id='T422_EARLY_FEEDBACK_RECOVERY_OVERNIGHT',out=str(OUT),root_out=str(OUT),phase='protect',student_start_model=str(model),student_start_sha256=sha(model),student_policy_hash=old['student_policy_hash'],job=old['job'],source_mixed=old['out']+'/mixed.npz',
        reference_path=str(reference),reference_sha256=sha(reference),initial_fixture_rows=fixtures,
        search_modes=['time','progress'],search_generations=3,search_population=12,search_elite=4,search_seed=20261107422,
        gain_mu=[1.,1.,1.,1.,.2,.3],gain_sd=[.7,.6,.7,1.,.3,.3],gain_low=[0.,0.,-1.,0.,0.,0.],gain_high=[4.,3.,3.,5.,1.5,1.5],gain_min_sd=[.15]*6,
        recovery_prefixes=[17,20],search_seconds=5400,search_audit_seconds=900,search_physics=6000000,search_commands=150,
        learning_rounds=4,bc_steps=200,epochs=8,learning_main_seconds=7200,learning_seconds=5400,learning_audit_seconds=3600,learning_physics=25000000,learning_commands=620,
        audit_seconds=3600,backup_seconds=900,source_seconds=240,all_seconds=36000,minimum_cycle_seconds=600,
        learning_stop_absolute=stop-5400,native_stop_absolute=stop-3600,all_stop_absolute=stop,
        final_bytes=96*2**20,search_final_bytes=16*2**20,learning_final_bytes=76*2**20,root_metadata_bytes=4*2**20,
        night_physics_max=31000000,night_commands_max=780,seed=20261107423,query_weight=1.,
        old_weight=1.,broad_old_weight=1.,left_weight=.25,right_weight=.25,
        preflight_discarded_Adam=8,preflight_zero_physics_commands=10,
        science_scope='Matched-budget feedback teacher early3.4s, conditional execution-target supervision and fresh on-policy PPO with old-gradient priority; joint engineering pilot, not novel formula or safety certificate.',
        label_contract='Actual bounded executed controls only; full-history suffix squared/one-sided-boundary surrogate, no fabricated teacher latent; no external controls in PPO ratios.',
        scientific_chain='72 distinct feedback gain proposals across2 modes, each exact native replay; successful actual selected only.4 conditional iterations prefix17/20 alternation, fresh student states,200BC+shadow and eligible4fresh+4exact/8PPO+shadow.2 consecutive rejections stop. No dynamic grid/seed/round extensions.',
        offline='All science, condition branches, audits and manifests use local installed dependencies, no network/LLM/API needed. Remote upload optional only after sealed science.',
        acceptance='Keep all initial/retained success IDs; BC must repair at least1; PPO only after accepted BC andold8 restored. Complete samecandidate old275+wide54 then fresh16 then78 exact native model readbacks for six. No preimplemented7/8 launch.',
        no_restart_or_parameter_grid=True,narrow_paused=True,no_seven_eight=True,safe_T255_frozen=True)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(reference),str(ROOT/'stage/runs/t366_factual_history/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T366.json'),str(ROOT/'stage/docs/day_20261107/T421_FINDINGS.json'),p['source_mixed'],str(model)}|{x['path'] for x in fixtures}
    p['inputs']={f:sha(f) for f in sorted(paths)}
    used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    prior=read(ROOT/'stage/docs/day_20261107/T421_FINDINGS.json');p['prior_day_actual']=dict(physics=prior['day_cumulative_physics'],commands=prior['day_cumulative_commands'],main_wall_s=prior['day_main_wall_seconds'])
    assert p['prior_day_actual']['physics']+p['night_physics_max']<=50_000_000 and p['prior_day_actual']['commands']+p['night_commands_max']<=2600 and p['prior_day_actual']['main_wall_s']+p['search_seconds']+p['learning_main_seconds']<=20000
    assert time.time()+5400+900+7200+3600+900+240<stop
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=used,global_full_reserved=used+p['final_bytes'],night_physics_max=31000000,night_commands_max=780,own_bytes=p['final_bytes'],max_live_Adam=832,max_shadow_Adam=832,max_independent_audit_discarded_Adam=1664,preflight_discarded_Adam=8,zero_native_preflight_commands=10,search_commands=150,learning_commands=620,prior_day_actual=p['prior_day_actual'],full_day_physics_worst=p['prior_day_actual']['physics']+31000000,full_day_commands_worst=p['prior_day_actual']['commands']+780,full_day_main_seconds_worst=p['prior_day_actual']['main_wall_s']+12600,all_absolute_beijing='2026-11-08 08:30',learning_absolute_beijing='2026-11-08 07:00',native_absolute_beijing='2026-11-08 07:30',no_network_dependency_for_science=True))
    print('T422_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    import learn as LEARN
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checks=R.pure_checks();ref=R.load_reference(p['reference_path'],p['reference_sha256']);base=E.N.science(p);q,opt=E.learner(base,p);assert q.phash()==p['student_policy_hash'];initial=[(E.N.S.arrays(r['path']),r) for r in p['initial_fixture_rows']]
    g=E.build_groups(q,initial,p);g['corrections']=[E.checked_query(q,x,p) for x in p['query_sources']]
    x=torch.as_tensor(ref['arrays']['states']);g['corrections'].append(dict(x=x,b=q.base_actions(x),actions=torch.as_tensor(ref['arrays']['actions'],dtype=torch.float64),first=0,source_id='PREFLIGHT_EXISTING_REFERENCE_EXECUTION_FIXTURE_NOT_NEW_RECOVERY',executed_teacher=True,actual_complete_history=True,safe_goal_verified=True,no_teacher_latent=True,not_onpolicy=True))
    shadow,so=E.learner(base,p);before=E.L.state(q,opt);events=0
    def guard():assert time.time()<p['learning_stop_absolute']
    def event(where,k):
        nonlocal events
        if where=='AFTER':events+=1
    mature=None
    for j in range(4):
        a=LEARN.fit_bc(q,opt,g,p,guard,event,steps=1);b=LEARN.fit_bc(shadow,so,g,p,guard,event,steps=1)
        assert a==b and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
        if j==0:mature=E.L.state(q,opt)
        if j>=2:
            E.L.load(q,opt,mature);E.L.load(shadow,so,mature);assert digest(E.L.state(q,opt))==digest(mature)
    assert events==8
    for name in ['t422_common.py','reference.py','learn.py','control.py','pipeline.py','audit.py','runner.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage422_check_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    pref=OUT/'preflight_native';pref.mkdir();w=s=pool=None
    try:
        with (pref/'console.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            native,create,w=setup(p,pref/'direct');q=frozen(p)
            def cancel(physical=False):
                if physical:raise E.L.StageStop('T422 preflight no native steps')
            s=E.L.Session(p['job'],native,create,w,cancel,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')),q);h=snap_hash(s);snap=s.capture();s.restore(snap);assert h==snap_hash(s)
            assert E.equal(s.initial,ref['arrays']['initial_native_integration']) and E.equal(s.packet(),ref['arrays']['states'][0])
            try:s.step(owner=0);raise AssertionError('Missing cancellation')
            except E.L.StageStop:pass
            assert s.steps==0 and not s.rows;s.close();s=None;w.close();w=None
        import multiprocessing as mp
        counts=mp.get_context('spawn').RawArray('q',5);(pref/'pool').mkdir();E.set_out(pref/'pool');pool=E.Pool(dict(p,out=str(pref/'pool'),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
        ids=[v['pid'] for v in pool.ready];values=pool.request([dict(model=p['student_start_model'],job=p['job'],first=0,seed=None,cancel_probe=True)]*4);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
        pool.close();pool=None;assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    finally:
        if s:s.close()
        if w:w.close()
        if pool:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_clone_Adam=8,zero_physics_commands=10,pure_feedback=checks,actual_reference_initial_native_and120_exact=True,mature_Adam_shadow_rejection_and_interruption_rollback=True,all_actual_entry_globals=True,actual_four_worker_cancel_and_exit=True,no_new_success_label=True,reference_execution_loss_fixture_only=True));E.N.manifest(OUT)
    print('T422_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+5400+900+7200+3600+900+240<p['all_stop_absolute'];limit=min(p['all_seconds'],p['all_stop_absolute']-now-15)
    with (DOC/'NIGHT_CONSOLE.log').open('xb') as f:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T422','--why=Offline bounded Stage overnight research','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,absolute_stop=p['all_stop_absolute'],source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,no_old_job_restart=True));print('T422_ACTUALLY_LAUNCHED',child.pid,flush=True)
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
def child(file,args,seconds,deadline,log):
    global ACTIVE
    assert time.time()+min(seconds,60)<deadline
    with Path(log).open('xb') as f:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        end=min(time.time()+seconds,deadline-5)
        try:
            while ACTIVE.poll() is None:
                assert time.time()<end
                save(DOC/'NIGHT_STATUS.json',dict(state='RUNNING',child=file,args=args,child_pid=ACTIVE.pid,driver_pid=os.getpid(),epoch=time.time(),deadline=deadline,science_has_no_network_dependency=True));time.sleep(5)
            assert ACTIVE.returncode==0,(file,ACTIVE.returncode)
        except BaseException:stop_active();raise
        finally:ACTIVE=None
def stopped(folder):
    summary=read(folder/'summary.json');assert summary['all_sessions_closed']
    ids=[read(folder/'main_started.json')['pid']]
    if (folder/'workers.json').exists():ids+=read(folder/'workers.json').get('workers',[])
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    return ids
def driver():
    p=read(OUT/'protocol.json');verify(p);launch=read(OUT/'launch.json');start=time.time();deadline=launch['deadline'];state='FAILED';error=None;remote='PENDING';source='PENDING';absent=[];learned=False
    assert not (OUT/'driver_started.json').exists();save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def interrupt(signum,frame):raise TimeoutError('T422 absolute deadline/interruption '+str(signum))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupt)
    signal.alarm(max(1,int(deadline-time.time()-3)))
    try:
        child('control.py',[],p['search_seconds'],min(deadline-1800,p['native_stop_absolute']),DOC/'SEARCH.log');absent+=stopped(OUT/'search')
        child('audit.py',['search'],p['search_audit_seconds'],deadline-1500,DOC/'SEARCH_AUDIT.log');assert read(OUT/'search/audit.json')['state']=='PASS'
        if read(OUT/'search/summary.json')['result']['valid_recovery']:
            needed=p['learning_main_seconds']+p['learning_audit_seconds']+p['backup_seconds']+p['source_seconds']+120
            assert time.time()+needed<deadline and time.time()+p['learning_seconds']<p['learning_stop_absolute']
            child('pipeline.py',[],p['learning_main_seconds'],min(deadline-4500,p['native_stop_absolute']),DOC/'LEARNING.log');absent+=stopped(OUT/'learning')
            child('audit.py',['learning'],p['learning_audit_seconds'],deadline-1200,DOC/'LEARNING_AUDIT.log');assert read(OUT/'learning/audit.json')['state']=='PASS';learned=True
        state='COMPLETE_AUDITED_LOCAL'
    except BaseException:
        error=traceback.format_exc();stop_active();save(DOC/'NIGHT_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
    finally:
        stop_active()
        for d in [OUT/'search',OUT/'learning']:
            if (d/'summary.json').exists():
                try:absent+=stopped(d)
                except BaseException:state='FAILED_UNCLOSED_NATIVE';error=(error or '')+'\n'+traceback.format_exc()
        closed=state!='FAILED_UNCLOSED_NATIVE';save(OUT/'SEALED.json',dict(state=state,epoch=time.time(),all_scientific_processes_stopped=closed,native_pids=sorted(set(absent)),error=error,no_future_scientific_writes=True));E.N.manifest(OUT)
        if closed and time.time()+60<deadline:
            try:
                child('transport.py',['T422'],p['backup_seconds'],deadline-300,DOC/'BACKUP.log');remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T422.json')['state']
            except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),science_not_blocked=True))
        search=read(OUT/'search/summary.json') if (OUT/'search/summary.json').exists() else None;learning=read(OUT/'learning/summary.json') if (OUT/'learning/summary.json').exists() else None
        ledger=dict(state=state,finished_epoch=time.time(),elapsed_s=time.time()-start,search=search,learning=learning,remote=remote,source='PENDING',all_native_stopped=closed,seven_eight_entered=False,missing_not_failure_rate=True)
        save(DOC/'FINAL_REPORT.json',ledger);save(DOC/'NIGHT_STATUS.json',dict(state=state,epoch=time.time(),all_native_stopped=closed,remote=remote))
        text='# Stage 2026-11-07夜间真实结果\n\n'+f'状态：{state}；全部原生退出：{closed}；远端：{remote}。\n\n'+('搜索未找到真实恢复，依协议未进入学习。\n' if not learned else '条件学习已执行，结果与完整六米门槛以FINAL_REPORT.json原始计数为准。\n')+'\n七／八米未启动。缺测为null，不算失败率；训练教师成功不等于学生六米稳定。\n'
        (DOC/'FINAL_REPORT.md').write_text(text)
        try:
            status=ROOT/'stage/docs/RESEARCH_STATUS.md'
            if sha(status)==launch['research_status_sha256']:status.write_text('2026-11-07夜间T422有限链已结束，state='+state+'，remote='+remote+'；全部原生退出='+str(closed)+'。准确搜索／学习／live-shadow／门槛计数见overnight_20261107/FINAL_REPORT.json；没有自动7／8米。所有原始保留，缺测null，不晋升失败模型，窄暂停、Stage/Phase隔离。\n\n'+status.read_text())
            files=[str(DOC/'FINAL_REPORT.json'),str(DOC/'FINAL_REPORT.md'),str(status)]
            for n in ['release_receipt_T422.json','source_manifest_T422.json']:
                f=ROOT/'stage/docs/long_distance_20261101'/n
                if f.exists():files.append(str(f))
            subprocess.run(['git','add','--',*files],cwd=ROOT,check=True,timeout=20);subprocess.run(['git','commit','-m','Report bounded Stage T422 overnight outcome'],cwd=ROOT,check=True,timeout=30)
            if time.time()+60<deadline:child('transport.py',['source_sync'],p['source_seconds'],deadline-5,DOC/'SOURCE.log');source='EXACT_REMOTE_ROOT_VERIFIED'
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),no_science_restart=True))
        save(DOC/'CLOSEOUT.json',dict(state=state,epoch=time.time(),remote=remote,source=source,all_native_stopped=closed,all_absolute_deadline=p['all_stop_absolute']));signal.alarm(0)
        print('T422_FINISHED',state,remote,source,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
