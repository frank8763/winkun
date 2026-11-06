from recovery_common import *


def driver():
    p=read(OUT/'protocol.json');start=time.time();deadline=start+p['all_seconds'];current=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,all_deadline=deadline))
    def stop(*_):raise TimeoutError('T391 absolute deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(p['all_seconds']-2)
    def call(script,mode,seconds):
        nonlocal current
        state=script+'_'+mode;save(DOC/'T391_DRIVER_STATUS.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time()))
        with (DOC/('T391_'+state+'.log')).open('wb') as log:
            current=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),mode],cwd=ROOT,
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if mode=='main':
                child=OUT/('teacher' if script=='teacher.py' else 'student')
                save(child/'launch.json',dict(pid=current.pid,epoch=time.time(),root_protocol_sha256=sha(OUT/'protocol.json')))
            try:code=current.wait(timeout=min(seconds,max(1,int(deadline-time.time())-1)))
            except subprocess.TimeoutExpired:
                os.killpg(current.pid,signal.SIGTERM);time.sleep(.2)
                try:os.killpg(current.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                current.wait();raise
            assert code==0,(state,code)
            # Refuse an orphaned simulator/worker before independent finalization.
            groups=subprocess.check_output(['ps','-eo','pgid='],text=True).split()
            for _ in range(100):
                if str(current.pid) not in groups:break
                time.sleep(.05);groups=subprocess.check_output(['ps','-eo','pgid='],text=True).split()
            assert str(current.pid) not in groups,(state,'child process group still alive')
            current=None
    try:
        teacher=OUT/'teacher';teacher.mkdir()
        tp=dict(p['teacher'],source_protocol=p['source_protocol'],inputs=p['inputs'],prefix_holds=p['prefix_holds'],remaining_holds=p['remaining_holds'],absolute_deadline=deadline-1200)
        save(teacher/'protocol.json',tp)
        call('teacher.py','main',p['teacher']['main_seconds'])
        call('teacher.py','audit',p['teacher']['audit_seconds'])
        ts=read(teacher/'summary.json');children=['teacher'];ss=None
        if ts['teacher_safe_GOAL']:
            assert time.time()+p['student']['main_seconds']+p['student']['audit_seconds']+p['backup_seconds']<deadline
            student=OUT/'student';student.mkdir();(student/'episodes').mkdir()
            sp=dict(p['student'],source_protocol=p['source_protocol'],inputs=dict(p['inputs']),cached_initial_policy=p['source_parent_policy_hash'],
                    absolute_native_deadline=deadline-p['student']['audit_seconds']-p['backup_seconds'],
                    absolute_learning_deadline=min(time.time()+p['student']['learning_seconds'],deadline-1800))
            sp['inputs'].update({str(f):sha(f) for f in [teacher/'mixed.npz',teacher/'audit.json',teacher/'summary.json']})
            save(student/'protocol.json',sp);save(student/'preflight.json',dict(state='PASS_ROOT_CONTRACT_REUSED',root_preflight_sha256=sha(OUT/'preflight.json'),additional_Adam=0))
            call('student.py','main',p['student']['main_seconds'])
            call('student.py','audit',p['student']['audit_seconds'])
            ss=read(student/'summary.json');children.append('student')
        total=dict(physics=ts['physics']+(0 if ss is None else ss['physics']),
                   commands=ts['commands']+(0 if ss is None else ss['commands'])+read(OUT/'preflight.json')['geometry'])
        assert total['physics']<=p['combined_max_physics'] and total['commands']<=p['combined_max_commands']
        final=dict(state='COMPLETE_NO_SUCCESSFUL_RECOVERY_NO_STUDENT' if ss is None else 'COMPLETE_CONDITIONAL_STUDENT_EVALUATED',
                   teacher=ts,student=ss,completed_children=children,all_native_groups_stopped=True,actual=total,
                   new_RL_updates=0,seven_eight_entered=False,remote='PENDING_UNTIL_VERIFIED',elapsed_s=time.time()-start)
        save(OUT/'local_final_report.json',final);N.manifest(OUT)
        call('transport.py','T391',p['backup_seconds'])
        save(DOC/'T391_FINAL_REPORT.json',dict(state='COMPLETE_AUDIT_AND_REMOTE_VERIFIED',actual=total,teacher_success=ts['teacher_safe_GOAL'],
            student=None if ss is None else ss['results'],elapsed_s=time.time()-start,new_RL_updates=0,
            remote_receipt=str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T391.json')))
        save(DOC/'T391_DRIVER_STATUS.json',dict(state='COMPLETE',pid=os.getpid(),updated_epoch=time.time()))
    except BaseException:
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM);time.sleep(.2);os.killpg(current.pid,signal.SIGKILL);current.wait()
            except ProcessLookupError:pass
        save(DOC/'T391_DRIVER_ERROR.json',dict(error=traceback.format_exc(),pid=os.getpid(),elapsed_s=time.time()-start));raise
    finally:signal.alarm(0)


def launch():
    assert not (OUT/'launch.json').exists();p=read(OUT/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS'
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=1024**3
    with (DOC/'T391_driver.log').open('wb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--who=Stage-T391','--why=Bounded actual-student recovery experiment',
          'timeout','--signal=TERM','--kill-after=1',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],
          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],all_seconds=p['all_seconds']))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


if __name__=='__main__':globals()[sys.argv[1]]()
