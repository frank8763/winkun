"""One bounded local Stage chain; no service is needed for scientific work."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback,shutil

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261107';ACTIVE=None
def read(f):return json.loads(Path(f).read_text())
def save(f,value):
    f=Path(f);f.parent.mkdir(parents=True,exist_ok=True)
    temporary=f.with_name(f.name+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(f)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(d):return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
def plan():
    p=read(HERE/'protocol.json');out=Path(p['out'])
    assert (HERE/'protocol.json').read_bytes()==(out/'protocol.json').read_bytes()
    return p,out
def verify(p,out):
    for f,h in p['inputs'].items():assert sha(f)==h,('Changed registered input',f)
    assert used(ROOT/'stage/runs')<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert sum(f.stat().st_size for f in out.iterdir() if f.is_file())<=2*2**20
def deadlines():
    tz=ZoneInfo('Asia/Shanghai')
    return {key:datetime(2026,11,7,hour,minute,tzinfo=tz).timestamp() for key,hour,minute in [('learning',20,0),('native',20,20),('all',21,0)]}
def stopped(out):
    pids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        f=out/name
        if f.exists():
            value=read(f);pids.update(value[k] for k in ['pid','parent'] if type(value.get(k)) is int);pids.update(value.get('workers',[]))
    for pid in pids:assert not Path('/proc',str(pid)).exists(),('Live native process',pid)
    return sorted(pids)
def stop():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait(timeout=2)
def call(file,args,seconds,absolute,log):
    global ACTIVE
    remaining=min(seconds,absolute-time.time()-5);assert remaining>0,('No child wall reserve',file,args)
    with Path(log).open('xb') as stream:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*map(str,args)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:assert ACTIVE.wait(timeout=remaining)==0,(file,args)
        except BaseException:stop();raise
        finally:ACTIVE=None
def member_manifest(out):
    values=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(out/'local_manifest.json',dict(state='LOCAL_ALL_BYTES_SHA_VERIFIED_REMOTE_PENDING',files=values,epoch=time.time(),remote_verified=False))
    for row in values:assert sha(ROOT/row['path'])==row['sha256']

def driver():
    p,out=plan();verify(p,out);cut=deadlines();launch=read(out/'launch.json');start=time.time()
    assert not (out/'driver_started.json').exists()
    deadline=min(launch['deadline'],cut['all']);assert start+30<deadline
    def interrupted(signum,frame):raise TimeoutError('Evening Stage hard stop '+str(signum))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(deadline-start-3)))
    lock=(out/'.driver.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline,offline_science=True))
    prior=p['prior_day_actual'];records=[];query_sources=copy.deepcopy(p.get('query_sources',[]));current=p['student_start_model'];state='FAILED';reason='UNFINISHED';backups=[]
    final_reserve=len(p['phases'])*p['backup_seconds']+120+120
    def ledger(phase):
        completed=[x for x in records if x.get('summary')]
        total=dict(physics=prior['physics']+sum(x['summary']['physics'] for x in completed),commands=prior['commands']+8+sum(x['summary']['commands'] for x in completed),main_wall_s=prior['main_wall_s']+sum(x['summary']['elapsed_s'] for x in completed))
        assert total['physics']<=50_000_000 and total['commands']<=2600 and total['main_wall_s']<=20000
        value=dict(state=phase,pid=os.getpid(),updated_epoch=time.time(),deadline=deadline,day_total=total,records=records,current_research_model=current,safe_T255_promoted=False,query_sources=query_sources,backups=backups)
        save(out/'status.json',value);save(DOC/'T418_LEDGER.json',value);return total
    try:
        ledger('REGISTERED_CHAIN_START')
        for item in p['phases']:
            name=item['name'];phase=item['phase'];folder=out/name;total=ledger('ADMISSION_'+name)
            worst_physics=p['worst_phase_physics'][phase] if isinstance(p['worst_phase_physics'],dict) else p['worst_phase_physics']
            worst_commands=p['worst_phase_commands'][phase] if isinstance(p['worst_phase_commands'],dict) else p['worst_phase_commands']
            why=[];now=time.time()
            if total['physics']+worst_physics>50_000_000:why.append('NO_FULL_PHYSICS_RESERVE')
            if total['commands']+worst_commands>2600:why.append('NO_FULL_COMMAND_RESERVE')
            if total['main_wall_s']+p['main_seconds']>20000:why.append('NO_FULL_MAIN_WALL_RESERVE')
            if now+p['learning_seconds']+30>=cut['learning'] or now+p['main_seconds']+5>=cut['native'] or now+p['main_seconds']+p['audit_seconds']+120>=deadline-final_reserve:why.append('NO_FULL_TIME_AND_FINALIZATION_RESERVE')
            if used(ROOT/'stage/runs')+p['final_bytes']>2**30 or shutil.disk_usage(ROOT).free<40*2**30:why.append('NO_FULL_CAPACITY_RESERVE')
            if why:
                records.append(dict(name=name,phase=phase,state='NOT_STARTED',reasons=why,summary=None));ledger('SKIPPED_'+name);continue
            assert not folder.exists();folder.mkdir();(folder/'episodes').mkdir();(folder/'blocks').mkdir()
            child=copy.deepcopy(p);child.update(item);child.update(out=str(folder),phase=phase,student_start_model=current,student_start_sha256=sha(current),query_sources=copy.deepcopy(query_sources),root_out=str(out),root_protocol_sha256=sha(out/'protocol.json'),prior_day_actual=total)
            previous_completed=[v for v in records if v.get('summary')]
            if previous_completed:
                previous_joint=previous_completed[-1]['summary']['result']['final_joint']
                child.pop('initial_control_rows',None)
                child.update(initial_expected_policy_hash=previous_joint['policy_hash'],initial_expected_success_ids=previous_joint['success_ids'])
            child['inputs'][current]=sha(current)
            for query in query_sources:
                for file in [query['row']['path'],query['source_student_model']]:child['inputs'][file]=sha(file)
            for previous in records:
                if previous.get('summary'):
                    f=out/previous['name']/'local_manifest.json';child['inputs'][str(f)]=sha(f)
            save(folder/'protocol.json',child);record=dict(name=name,phase=phase,state='RUNNING',protocol_sha256=sha(folder/'protocol.json'),start_epoch=time.time(),summary=None);records.append(record);ledger('NATIVE_'+name)
            call('run.py',[folder/'protocol.json'],p['main_seconds']+2,cut['native'],DOC/('T418_'+name+'_run.log'))
            summary=read(folder/'summary.json');record['summary']=summary;assert summary['state']=='COMPLETE'
            record['native_pids_absent']=stopped(folder);ledger('AUDIT_'+name)
            call('audit.py',[folder/'protocol.json'],p['audit_seconds'],deadline-final_reserve,DOC/('T418_'+name+'_audit.log'))
            assert read(folder/'audit.json')['state']=='PASS';record['native_pids_absent']=stopped(folder)
            member_manifest(folder);record.update(state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING',manifest_sha256=sha(folder/'local_manifest.json'))
            model=summary['result']['model'];assert sha(model['path'])==model['sha256'];current=model['path']
            qpath=folder/'query_sources.json'
            if qpath.exists():
                values=read(qpath);assert isinstance(values,list)
                known={(q['row']['path'],q['row']['sha256']) for q in query_sources}
                for query in values:
                    key=query['row']['path'],query['row']['sha256']
                    assert query['row']['success'] and sha(query['row']['path'])==key[1] and sha(query['source_student_model'])==query['source_student_sha256']
                    if key not in known:query_sources.append(query);known.add(key)
            ledger('COMPLETE_'+name)
            if summary['result']['stable_wide_six']:
                reason='FULL_SIX_GATE_REACHED_NO_UNREGISTERED_SEVEN_IMPLEMENTATION';break
        else:reason='ALL_PREDECLARED_PHASES_COMPLETED_OR_BOUNDED_SKIPPED'
        state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING';ledger('SCIENTIFIC_CHAIN_FINISHED')
        # Scientific child outputs are stopped and frozen before any network
        # transport. An outage cannot prevent a preprogrammed next mechanism.
        for record in records:
            if record['state']!='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING':continue
            name=record['name'];group='T418-'+name;receipt=ROOT/'stage/docs/long_distance_20261101'/('release_receipt_'+group+'.json')
            try:
                if not receipt.exists():call('transport.py',[group],300,deadline-180,DOC/('T418_'+name+'_backup.log'))
                value=read(receipt);assert value['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
                backups.append(dict(name=name,state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',receipt=str(receipt),sha256=value['sha256']));record['state']='COMPLETE_AUDITED_REMOTE_VERIFIED'
            except BaseException:backups.append(dict(name=name,state='REMOTE_PENDING',error=traceback.format_exc()))
            ledger('BACKUP_'+name)
        completed=[x for x in records if x.get('summary')]
        if completed and all(x['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED' for x in completed):state='COMPLETE_AUDITED_REMOTE_VERIFIED'
    except BaseException:
        stop();save(DOC/'T418_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),no_training_restart=True));reason='HARD_ERROR_OR_DEADLINE_STOP'
        for record in records:
            f=out/record['name']/'summary.json'
            if record['state']=='RUNNING' and f.exists():record['summary']=read(f);record['state']='FAILED_RAW_PRESERVED'
    finally:
        stop();signal.alarm(0)
        totals=ledger(state)
        report=dict(state=state,pid=os.getpid(),finished_epoch=time.time(),elapsed_s=time.time()-start,deadline=deadline,reason=reason,day_total=totals,records=records,backups=backups,current_research_model=current,safe_T255_promoted=False,seven_eight_entered=False,narrow_paused=True)
        save(DOC/'T418_FINAL_REPORT.json',report);save(out/'final_report.json',report)
        (DOC/'T418_FINDINGS.md').write_text(f"# T418晚间后台真实结果\n\n状态：{state}。停止理由：{reason}。\n\n日累计物理{totals['physics']}，命令{totals['commands']}，主管线{totals['main_wall_s']:.3f}秒。按原始每phase summary/audit和FINAL_REPORT逐项报告；NOT_STARTED是预算/时钟未准入，不计失败率。研究模型不替换safeT255。七米/八米未进入，窄场景暂停。\n\n详细机器可读报告：[T418_FINAL_REPORT.json](T418_FINAL_REPORT.json)。必要二进制仅在完整远端逐成员SHA后称备份完成；pending明确保留。\n")
        member_manifest(out)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    # Final source publication is separate from the immutable science chain.
    # The latest root-tree receipt is the source-of-truth after this commit.
    try:
        assert time.time()+120<deadline
        status_path=ROOT/'stage/docs/RESEARCH_STATUS.md'
        facts=[]
        for record in records:
            if record.get('summary'):
                summary=record['summary'];result=summary.get('result',{})
                joint=result.get('final_joint',{})
                facts.append(dict(name=record['name'],state=record['state'],physics=summary['physics'],commands=summary['commands'],updates=summary['updates'],final_joint=joint,stable_wide_six=result.get('stable_wide_six',False),main54=result.get('main54'),old275=result.get('old275'),fresh16=result.get('fresh16'),reload78=result.get('reload78')))
            else:facts.append(dict(name=record['name'],state=record['state'],reasons=record.get('reasons',[])))
        line='2026-11-07 T418晚间固定链已收尾，state='+state+'，reason='+reason+'；日累计物理/命令/主管秒='+json.dumps(totals,ensure_ascii=False)+'。逐项真实结果='+json.dumps(facts,ensure_ascii=False)+'。缺测null不是失败率；研究副本不晋升替换safeT255，窄暂停，七八米未进入。全部有限机制结束不无限续，备份以FINAL_REPORT逐项remote状态为准；根树以latest_source_sync_receipt为准。\n\n'
        status_path.write_text(line+status_path.read_text())
        files=[str(f.relative_to(ROOT)) for pattern in ['T418*.json','T418*.md'] for f in DOC.glob(pattern)]
        files.append('stage/docs/RESEARCH_STATUS.md')
        for pattern in ['release_receipt_T418-*.json','source_manifest_T418-*.json']:files.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob(pattern))
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Record bounded evening Stage repair chain outcomes'],cwd=ROOT,check=True)
        call('transport.py',['source_sync'],120,deadline,DOC/'T418_source_sync.log')
    except BaseException:save(DOC/'T418_SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),no_science_restart=True))
    print(json.dumps(dict(state=state,reason=reason,day_total=totals)),flush=True)

def launch():
    p,out=plan();verify(p,out);cut=deadlines();assert read(out/'preflight.json')['state']=='PASS' and not (out/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['root_tree']==tree
    now=time.time();assert now+p['main_seconds']+p['audit_seconds']+len(p['phases'])*p['backup_seconds']+360<cut['all'] and now+p['learning_seconds']+30<cut['learning']
    with (DOC/'T418_CONSOLE.log').open('xb') as log:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-Evening2100','--why=Bounded local Stage repair chain until21:00','timeout','--signal=TERM','--kill-after=2',str(max(1,int(cut['all']-now))),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'launch.json',dict(pid=process.pid,launcher_pid=os.getpid(),epoch=now,deadline=cut['all'],source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(out/'protocol.json'),hard_all_stop_Beijing='2026-11-07T21:00:00+08:00',offline_science=True))
    print(json.dumps(dict(state='ACTUALLY_LAUNCHED',pid=process.pid,deadline=cut['all'])),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
