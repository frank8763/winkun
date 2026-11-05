"""Independent offline full-task search and valid-source student chain."""
from night_common import *
ACTIVE=None
def terminate():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait(timeout=3)
def call(script,mode,out,seconds,log):
    global ACTIVE
    with log.open('xb') as f:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),mode,str(out)],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(DOC/'CURRENT_PROCESS.json',dict(pid=ACTIVE.pid,mode=mode,script=script,directory=str(out),epoch=time.time()))
        try:assert ACTIVE.wait(timeout=seconds)==0,(script,mode)
        except BaseException:terminate();raise
        finally:ACTIVE=None
def launch():
    p=read(HERE/'protocol.json');assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time();remaining=p['all_deadline']-epoch;assert remaining>25260
    with (OUT/'driver.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage offline full-task search and student learning','timeout','--signal=TERM','--kill-after=2s',str(int(remaining))+'s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'main'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=epoch,source_local_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],deadline=p['all_deadline'],network_required=False))
    save(DOC/'NIGHT_LAUNCH.json',read(OUT/'launch.json'));print(json.dumps(dict(state='FULL_TASK_OFFLINE_CHAIN_LAUNCHED',pid=proc.pid)))
def main():
    offline();p=read(HERE/'protocol.json');assert not (OUT/'driver_started.json').exists();start=time.time();deadline=p['all_deadline']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,external_network_blocked=True))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Night stop signal')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('08:30 absolute stop')));signal.alarm(max(1,int(deadline-start)-5))
    ledger=dict(state='RUNNING_OFFLINE_FULL_TASK_CHAIN',pid=os.getpid(),start_epoch=start,completed=[],current=None,actual=copy.deepcopy(p['prior_actual']),remote='PENDING_OFFLINE_NO_UPLOAD_ATTEMPT',seven_eight=False,all_deadline=deadline)
    teacher=None;complete=False
    def update():
        assert K.S.used(ROOT/'stage/runs')<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
        save(DOC/'NIGHT_LEDGER.json',ledger);save(DOC/'NIGHT_STATUS.json',dict(state=ledger['state'],current=ledger['current'],updated_epoch=time.time(),driver_pid=os.getpid(),remote=ledger['remote']))
    def run_stage(folder,spec,script,audit_seconds):
        out=OUT/folder;assert not out.exists();now=time.time()
        for f,h in p['inputs'].items():assert sha(f)==h,f
        assert now+spec['main_seconds']+audit_seconds+120<deadline and now+spec['main_seconds']<p['native_deadline'],'Full worst-case time reserve unavailable'
        assert K.S.used(ROOT/'stage/runs')+spec['final_bytes']<1024**3,'Full storage reserve unavailable'
        for key in ['physics','commands']:assert ledger['actual'][key]+spec['max_'+key]<=p['limits'][key]
        out.mkdir();(out/'episodes').mkdir() if script=='student.py' else None
        stagep=dict(spec,source_protocol=p['source_protocol'],native_deadline=min(now+spec['main_seconds'],p['native_deadline']),learning_deadline=p['learning_deadline'],inputs=p['inputs'])
        if script=='student.py':
            stagep['teacher']=None if teacher is None else str(teacher)
            stagep['learning_source']='VERIFIED_HUMAN_ONLY_CAPACITY_CONTROL' if teacher is None else 'FULL_SUCCESS_NATIVE_TEACHER'
            stagep['teacher_input_hashes']={} if teacher is None else {str(f):sha(f) for f in [teacher/'mixed.npz',teacher/'summary.json',teacher/'audit.json',teacher/'local_manifest.json']}
        save(out/'protocol.json',stagep);save(out/'launch.json',dict(epoch=now,root_protocol_sha256=sha(OUT/'protocol.json'),source_local_commit=read(OUT/'launch.json')['source_local_commit'],source_remote_commit=read(OUT/'launch.json')['source_remote_commit'],learning_source=stagep.get('learning_source'),input_teacher=stagep.get('teacher'),network_required=False))
        ledger['current']=dict(folder=folder,phase='MAIN');update()
        call(script,'main',out,min(spec['main_seconds']+5,p['native_deadline']-time.time()),out/'main.log')
        s=read(out/'summary.json');stopped(out);ledger['actual']['physics']+=s['physics'];ledger['actual']['commands']+=s['commands'];update()
        assert s['state']=='COMPLETE','Shared numerical/engineering failure stops chain'
        ledger['current']['phase']='INDEPENDENT_AUDIT';update();call(script,'audit',out,min(audit_seconds,deadline-time.time()-120),out/'audit_console.log');assert read(out/'audit.json')['state']=='PASS';manifest(out)
        ledger['completed'].append(dict(folder=folder,state='LOCAL_AUDIT_MANIFEST_PASS_REMOTE_PENDING',summary=s,manifest_sha256=sha(out/'local_manifest.json')));ledger['current']=None;update();return out,s
    try:
        update()
        for spec in p['search_specs']:
            out,s=run_stage(spec['folder'],dict(p['search'],**spec),'search.py',300)
            if s['teacher_safe_GOAL']:teacher=out;break
        ledger['student_source']='VERIFIED_HUMAN_ONLY_CAPACITY_CONTROL' if teacher is None else 'FULL_SUCCESS_NATIVE_TEACHER'
        if time.time()+3600>=p['learning_deadline']:ledger['student']='NOT_RUN_LEARNING_RESERVE'
        else:run_stage('student',p['student'],'student.py',900);ledger['student']='EXECUTED_ONE_FIXED_FIT_AND_NATIVE_EVALUATION'
        complete=True;ledger['state']='FINITE_FULL_TASK_CHAIN_COMPLETE'
    except BaseException:
        terminate();ledger['state']='STOPPED_WITH_ERROR_OR_HARD_BUDGET'
        save(DOC/'NIGHT_ERROR.json',dict(error=traceback.format_exc(),current=ledger['current'],epoch=time.time(),restarted=False))
        if ledger['current']:
            out=OUT/ledger['current']['folder']
            if out.exists():manifest(out)
    finally:
        terminate();signal.alarm(0);ledger.update(end_epoch=time.time(),elapsed_s=time.time()-start,complete=complete,all_child_groups_stopped=True,teacher_selected=None if teacher is None else str(teacher));update()
        save(DOC/'FINAL_REPORT.json',ledger)
        lines=['# 11月5日第二条离线链真实结果',f'状态：{ledger["state"]}。所有阶段依赖本地固定代码，无网络/API/LLM。',f'含前链实际的总物理={ledger["actual"]["physics"]}，命令={ledger["actual"]["commands"]}。',f'学生来源={ledger.get("student_source","未执行")}; 学生状态={ledger.get("student","未执行/缺测")}。','CEM是仿真动作序列搜索，学生是监督学习；本链新RL更新0。教师GOAL不是Actor自主六米通过。原场景完整门缺测null，7/8未实现未进入，远端备份PENDING。']
        for rec in ledger['completed']:
            s=rec['summary'];lines.append(f'- {rec["folder"]}: {s["state"]},physics={s["physics"]},commands={s["commands"]},teacherGOAL={s.get("teacher_safe_GOAL","不适用")}')
        (DOC/'FINAL_REPORT.md').write_text('\n\n'.join(lines)+'\n');manifest(OUT)
if __name__=='__main__':globals()[sys.argv[1]]()
