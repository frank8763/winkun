"""Single offline writer for the bounded Stage daytime repair chain.

Every scientific branch and condition is registered before launch.  Network
transport starts only after the complete scientific tree is stopped and sealed.
Missing summaries are interruptions, never zero-success completed experiments.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,time,json,hashlib,signal,subprocess,traceback,shutil,fcntl
import t426_core as C

ROOT,HERE,OUT,DOC=C.ROOT,C.HERE,C.OUT,C.DOC
read,save,sha=C.read,C.save,C.sha
STAGES=[('data','data'),('learn','learn'),('gate','gate'),('eight','eight')]
ACTIVE=None
ACTIVE_PGID=None
ACTIVE_STARTED=None


def used(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob('*') if path.is_file())


def hard_deadlines(p,launch):
    zone=ZoneInfo('Asia/Shanghai')
    fixed={name:datetime(2026,11,8,hour,0,tzinfo=zone).timestamp()
           for name,hour in [('learning',20),('native',20),('all',21)]}
    fixed['native']+=30*60
    return dict(learning=min(p['deadline_learning'],fixed['learning']),
                native=min(p['deadline_native'],fixed['native']),
                all=min(p['deadline_all'],fixed['all'],launch['deadline']))


def root_limits(p):
    values=p['root_limits']
    physics=values.get('max_physics',values.get('physics',25_400_000))
    commands=values.get('max_commands',values.get('commands',1100))
    size=values.get('max_bytes',values.get('final_bytes',values.get('bytes',96*2**20)))
    assert 0<physics<=25_400_000 and 0<commands<=1100 and 0<size<=96*2**20
    return dict(physics=physics,commands=commands,bytes=size)


def process_info(pid):
    """Read only non-secret process identity and process-group fields."""
    folder=Path('/proc',str(pid))
    try:
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=fields[0],parent=int(fields[1]),
                    pgrp=int(fields[2]),start_ticks=int(fields[19]))
    except (FileNotFoundError,ProcessLookupError):return None


def group_members(pgid):
    result=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        row=process_info(int(folder.name))
        if row is not None and row['pgrp']==pgid and row['state']!='Z':result.append(row)
    return result


def stop_active():
    """Clean the whole child group even if its main process already exited."""
    global ACTIVE,ACTIVE_PGID,ACTIVE_STARTED
    if ACTIVE_PGID is None:return []
    group=ACTIVE_PGID
    for sig,seconds in [(signal.SIGTERM,3.),(signal.SIGKILL,2.)]:
        members=group_members(group)
        if not members:break
        # A newly recycled unrelated group must not be killed on numeric PID
        # alone.  Real descendants cannot predate this launched child.
        if ACTIVE_STARTED is not None:
            assert all(row['start_ticks']>=ACTIVE_STARTED for row in members),('Reused child group',group)
        try:os.killpg(group,sig)
        except ProcessLookupError:pass
        stop=time.monotonic()+seconds
        while group_members(group) and time.monotonic()<stop:time.sleep(.05)
    if ACTIVE is not None:
        try:ACTIVE.wait(timeout=.5)
        except subprocess.TimeoutExpired:pass
    remaining=group_members(group)
    ACTIVE=None;ACTIVE_PGID=None;ACTIVE_STARTED=None
    return remaining


def native_pids(folder):
    """A missing summary never hides a started native worker tree."""
    folder=Path(folder);ids=set();started=False
    for name in ['main_started.json','workers.json','summary.json']:
        file=folder/name
        if not file.exists():continue
        row=read(file)
        if name=='main_started.json':started=True
        ids.update(row[key] for key in ['pid','parent','main_pid'] if type(row.get(key)) is int)
        ids.update(row.get('workers',[]))
    active=[process_info(pid) for pid in sorted(ids) if process_info(pid) is not None]
    return dict(started=started,pids=sorted(ids),live=active,
                summary_exists=(folder/'summary.json').exists(),
                all_native_processes_stopped=not active)


def check_stopped(folder,require_complete=False):
    evidence=native_pids(folder)
    assert evidence['started'] and evidence['pids'],('No native start evidence',str(folder))
    assert evidence['all_native_processes_stopped'],('Native process remains or PID reused',evidence)
    if require_complete:
        summary=read(Path(folder)/'summary.json')
        assert summary['state']=='COMPLETE'
        assert summary['all_workers_stopped'] and summary['all_sessions_closed']
    return evidence


def member_manifest(folder):
    folder=Path(folder)
    members=[dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=sha(file))
             for file in sorted(folder.rglob('*'))
             if file.is_file() and file.name!='local_manifest.json']
    save(folder/'local_manifest.json',dict(state='LOCAL_ALL_BYTES_SHA_VERIFIED_REMOTE_PENDING',
         files=members,epoch=time.time(),network_required=False,remote_verified=False))
    for row in members:
        file=ROOT/row['path']
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
    return sha(folder/'local_manifest.json')


def check_manifest(folder):
    file=Path(folder)/'local_manifest.json';members=read(file)['files']
    for row in members:
        path=ROOT/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    return sha(file)


def call(file,args,seconds,deadline,log,status,heartbeat=True):
    global ACTIVE,ACTIVE_PGID,ACTIVE_STARTED
    assert ACTIVE is None and ACTIVE_PGID is None
    remaining=min(float(seconds),deadline-time.time()-5)
    assert remaining>0,('No complete child window',file,args)
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('xb') as stream:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*map(str,args)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE_PGID=ACTIVE.pid
        identity=process_info(ACTIVE.pid);ACTIVE_STARTED=None if identity is None else identity['start_ticks']
        end=time.time()+remaining
        print(json.dumps(dict(event='CHILD_STARTED',file=file,args=list(map(str,args)),pid=ACTIVE.pid,
                              pgrp=ACTIVE_PGID,deadline=end)),flush=True)
        try:
            while ACTIVE.poll() is None:
                assert time.time()<end,('Child relative/absolute deadline',file,args)
                if heartbeat:
                    save(DOC/'DRIVER_STATUS.json',dict(state=status,epoch=time.time(),driver_pid=os.getpid(),
                        child_pid=ACTIVE.pid,child_pgrp=ACTIVE_PGID,child_file=file,args=list(map(str,args)),
                        child_deadline=end,offline_science=True))
                time.sleep(min(5.,max(.05,end-time.time())))
            code=ACTIVE.returncode
            remaining_group=stop_active()
            assert not remaining_group,('Child descendants remained',file,remaining_group)
            assert code==0,(file,args,code)
            print(json.dumps(dict(event='CHILD_FINISHED',file=file,args=list(map(str,args)),exit_code=code)),flush=True)
        except BaseException:
            remaining_group=stop_active()
            if remaining_group:save(DOC/'UNCLOSED_CHILD_GROUP.json',dict(epoch=time.time(),file=file,members=remaining_group))
            raise


def main():
    p=read(OUT/'protocol.json');C.verify(p)
    # Popen returns before the parent atomically publishes the launch receipt.
    wait_until=time.monotonic()+10.
    while not (OUT/'launch.json').exists():
        assert time.monotonic()<wait_until,'Missing actual launch receipt'
        time.sleep(.05)
    launch=read(OUT/'launch.json');cut=hard_deadlines(p,launch);limit=root_limits(p)
    assert not (OUT/'driver_started.json').exists()
    assert time.time()+60<cut['all']
    started=time.time();state='FAILED';reason='UNFINISHED';records=[];error=None
    remote='REMOTE_PENDING';source='SOURCE_PENDING';six_gate=False;eight_started=False
    final_reserve=float(p.get('backup_seconds',600))+float(p.get('source_seconds',180))+120.
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=started,
        deadline=cut['all'],learning_deadline=cut['learning'],native_deadline=cut['native'],offline_science=True))
    lock=(OUT/'.driver.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupt(signum,frame):raise TimeoutError('T426 registered hard stop or interruption '+str(signum))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupt)
    signal.alarm(max(1,int(cut['all']-time.time()-30)))
    def totals():
        known=[row['summary'] for row in records if row.get('summary') is not None]
        pref=p.get('preflight_zero_physics_commands',p.get('preflight_zero_commands',0))
        return dict(physics=sum(row['physics'] for row in known),commands=pref+sum(row['commands'] for row in known),
                    main_wall_s=sum(row['elapsed_s'] for row in known))
    def ledger(stage):
        actual=totals()
        assert actual['physics']<=limit['physics'] and actual['commands']<=limit['commands']
        assert used(OUT)<=limit['bytes'] and used(ROOT/'stage/runs')<=2**30
        assert shutil.disk_usage(ROOT).free>=40*2**30
        save(DOC/'DAY_LEDGER.json',dict(state=stage,epoch=time.time(),driver_pid=os.getpid(),deadline=cut,
             actual=actual,records=records,six_gate=six_gate,eight_started=eight_started,
             source=source,remote=remote,all_science_offline=True))
        return actual
    try:
        ledger('REGISTERED_CHAIN_START')
        for name,alias in STAGES:
            folder=OUT/alias;budget=p['stage_budgets'][name];actual=ledger('ADMISSION_'+name)
            complete=[row for row in records if row['state']=='COMPLETE_AUDITED_LOCAL']
            why=[]
            if name=='learn':
                data=next((row for row in complete if row['name']=='data'),None)
                if data is None:why.append('NO_COMPLETE_AUDITED_DATA_STAGE')
                elif data['summary']['result'].get('learning_data_available') is False:why.append('NO_VERIFIED_LEARNING_DATA')
            if name=='gate':
                if not any(row['name']=='learn' and row['summary']['result'].get('development_all_safe') is True for row in complete):
                    why.append('NO_ACCEPTED_FULL_329_DEVELOPMENT_MODEL')
            if name=='eight' and not six_gate:why.append('FULL_SIX_GATE_NOT_PASSED')
            now=time.time()
            if actual['physics']+budget['max_physics']>limit['physics']:why.append('NO_FULL_PHYSICS_RESERVE')
            if actual['commands']+budget['max_commands']>limit['commands']:why.append('NO_FULL_COMMAND_RESERVE')
            if used(OUT)+budget['max_bytes']>limit['bytes'] or used(ROOT/'stage/runs')+budget['max_bytes']>2**30 or shutil.disk_usage(ROOT).free<40*2**30:
                why.append('NO_FULL_CAPACITY_RESERVE')
            if now+budget['main_seconds']+5>=cut['native']:why.append('NO_FULL_NATIVE_TIME_RESERVE')
            if name=='learn' and now+budget['main_seconds']+5>=cut['learning']:
                why.append('NO_FULL_LEARNING_TIME_RESERVE')
            needed=budget['main_seconds']+budget['audit_seconds']+120+final_reserve
            if now+needed>=cut['all']:why.append('NO_FULL_AUDIT_AND_FINALIZATION_RESERVE')
            if why:
                records.append(dict(name=name,folder=str(folder),state='NOT_STARTED',reasons=why,summary=None))
                ledger('SKIPPED_'+name);continue
            assert not (folder/'main_started.json').exists() and not (folder/'summary.json').exists(),('Refuse duplicate stage',name)
            record=dict(name=name,folder=str(folder),state='RUNNING',start_epoch=time.time(),summary=None)
            records.append(record)
            if name=='eight':eight_started=True
            ledger('NATIVE_'+name)
            call('run.py',[name],budget['main_seconds'],cut['native'],DOC/(name+'_MAIN.log'),'NATIVE_'+name)
            record['native_stop_evidence']=check_stopped(folder,require_complete=True)
            summary=read(folder/'summary.json');record['summary']=summary
            assert summary['physics']<=budget['max_physics'] and summary['commands']<=budget['max_commands']
            assert summary['elapsed_s']<=budget['main_seconds']
            ledger('AUDIT_'+name)
            call('audit.py',[name],budget['audit_seconds'],cut['all']-final_reserve,
                 DOC/(name+'_AUDIT.log'),'AUDIT_'+name)
            assert read(folder/'audit.json')['state']=='PASS'
            record['native_stop_evidence']=check_stopped(folder,require_complete=True)
            record['manifest_sha256']=check_manifest(folder)
            record['state']='COMPLETE_AUDITED_LOCAL'
            if name=='gate':six_gate=summary['result'].get('six_gate') is True
            ledger('COMPLETE_'+name)
        state='COMPLETE_AUDITED_LOCAL'
        reason='ALL_PREDECLARED_STAGES_COMPLETED_OR_BOUNDED_SKIPPED'
    except BaseException:
        error=traceback.format_exc();remaining=stop_active()
        if remaining:state='FAILED_UNCLOSED_CHILD_GROUP'
        reason='CONTRACT_FAILURE_INTERRUPTION_OR_HARD_DEADLINE'
        save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_science_restart=True))
        for record in records:
            if record['state']!='RUNNING':continue
            file=Path(record['folder'])/'summary.json'
            record['summary']=read(file) if file.exists() else None
            record['state']='FAILED_RAW_PRESERVED' if file.exists() else 'INTERRUPTED_NO_SUMMARY_RAW_PRESERVED'
    finally:
        remaining=stop_active();all_native=[];closed=not remaining
        for name,alias in STAGES:
            folder=OUT/alias
            if not folder.exists():continue
            evidence=native_pids(folder);all_native.append(dict(name=name,**evidence))
            if evidence['started'] and not evidence['all_native_processes_stopped']:closed=False
        if not closed:state='FAILED_UNCLOSED_NATIVE_TREE'
        # No root run file is modified after this seal/manifest.  Backup may
        # therefore include a stopped root while the driver publishes docs.
        save(OUT/'SEALED.json',dict(state=state,reason=reason,epoch=time.time(),
             all_scientific_processes_stopped=closed,native_trees=all_native,
             error=error,six_gate=six_gate,eight_started=eight_started,
             no_future_scientific_writes=True,missing_summary_is_interruption_not_completed_failure=True))
        member_manifest(OUT)
        actual=ledger('SCIENTIFIC_CHAIN_SEALED')
        if closed and time.time()+60<cut['all']:
            try:
                receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T426.json'
                if not receipt.exists():call('transport.py',['T426'],p.get('backup_seconds',600),cut['all']-240,DOC/'BACKUP.log','OPTIONAL_REMOTE_BACKUP')
                value=read(receipt);assert value['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
                remote=value['state']
            except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),all_science_already_finished=True))
        report=dict(state=state,reason=reason,finished_epoch=time.time(),elapsed_s=time.time()-started,
            deadline=cut,records=records,actual=actual,all_scientific_processes_stopped=closed,
            native_trees=all_native,six_gate=six_gate,seven_skipped_by_explicit_user_request=True,
            eight_started=eight_started,remote=remote,source='CHECK_LATEST_SOURCE_SYNC_RECEIPT_AFTER_FINAL_COMMIT',
            error=error,safe_T255_not_promoted=True,narrow_paused=True,
            no_missing_summary_or_unexecuted_group_fabricated=True)
        save(DOC/'FINAL_REPORT.json',report)
        (DOC/'FINAL_REPORT.md').write_text('# Stage 2026-11-08离线后台真实结果\n\n'+
            f'状态：{state}。停止理由：{reason}。\n\n'+
            f'已记录物理{actual["physics"]}、命令{actual["commands"]}、主流程{actual["main_wall_s"]:.3f}秒。'+
            f'六米完整门槛={six_gate}；按用户授权跳过七米；八米实际启动={eight_started}。\n\n'+
            '逐支线实际结果和未启动理由以FINAL_REPORT.json及原始summary/audit为准；缺测null不是失败率。'+
            '研究模型不替换safeT255，窄场景暂停。历史T422训练复现缺口以独立记录为准，不由本链结果抹去。\n\n'+
            f'全部科学进程退出={closed}；远端状态={remote}。联网失败仅标PENDING，本地科学不依赖网络。\n')
        save(DOC/'DRIVER_STATUS.json',dict(state=state,epoch=time.time(),driver_pid=os.getpid(),
             all_scientific_processes_stopped=closed,six_gate=six_gate,eight_started=eight_started,remote=remote))
        # Source publication is optional and strictly later than scientific
        # sealing.  Do not mutate a changed user/research status document.
        try:
            assert closed and time.time()+90<cut['all']
            status=ROOT/'stage/docs/RESEARCH_STATUS.md'
            previous_hash=launch.get('research_status_sha256')
            status_updated=False
            if previous_hash is not None and sha(status)==previous_hash:
                status.write_text('2026-11-08 T426离线有限链结束，state='+state+'；six_gate='+str(six_gate)+
                    '，用户明确跳过7m，eight_started='+str(eight_started)+'。全部科学退出='+str(closed)+
                    '；remote='+remote+'。真实各支线/预算/缺测见evening_20261108/FINAL_REPORT.json，'+
                    '不复跑结束目录、不晋升safeT255、窄暂停、Stage/Phase隔离。\n\n'+status.read_text())
                status_updated=True
            files=[DOC/'FINAL_REPORT.json',DOC/'FINAL_REPORT.md',DOC/'DAY_LEDGER.json',DOC/'DRIVER_STATUS.json']
            if status_updated:files.append(status)
            for name in ['release_receipt_T426.json','source_manifest_T426.json','T426_small_parts_progress.json']:
                path=ROOT/'stage/docs/long_distance_20261101'/name
                if path.exists():files.append(path)
            subprocess.run(['git','add','--',*map(str,files)],cwd=ROOT,check=True,timeout=20)
            subprocess.run(['git','commit','-m','Report bounded Stage T426 offline daytime outcomes'],cwd=ROOT,check=True,timeout=30)
            call('transport.py',['source_sync'],p.get('source_seconds',180),cut['all']-5,DOC/'SOURCE.log','OPTIONAL_EXACT_SOURCE_SYNC',heartbeat=False)
            source='EXACT_REMOTE_ROOT_VERIFIED'
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),science_not_restarted=True))
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        print(json.dumps(dict(state=state,reason=reason,all_native_stopped=closed,six_gate=six_gate,
                              eight_started=eight_started,remote=remote,source=source)),flush=True)


if __name__=='__main__':main()
