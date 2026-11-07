"""Unique bounded joint-route producer, auditor and evidence finalizer.

No source synchronization runs while a native producer or worker is alive.
Only this experiment's small outcome documents enter the final source commit.
"""
from t408_core import *

ACTIVE=None

def install_handlers():
    def interrupted(signum,frame):
        raise TimeoutError('T408 driver interrupted '+str(signum))
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:
        signal.signal(signum,interrupted)
    return interrupted

def stop():
    global ACTIVE
    if ACTIVE and ACTIVE.poll() is None:
        try:
            os.killpg(ACTIVE.pid,signal.SIGTERM)
            ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid,signal.SIGKILL)
            ACTIVE.wait()
        except ProcessLookupError:
            pass

def native_stopped():
    known=set()
    for name in ['main_started.json','workers.json','summary.json']:
        path=OUT/name
        if not path.exists():
            continue
        value=read(path)
        known.update(value[key] for key in ['pid','parent'] if type(value.get(key)) is int)
        known.update(value.get('workers',[]))
    for pid in known:
        assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
    return sorted(known)

def final_source_commit():
    """Commit the exact small final-evidence allowlist, never unrelated edits."""
    native_stopped()
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip(),'Preexisting staged changes'
    candidates=list(DOC.glob('T408*.json'))+list(DOC.glob('T408*.md'))
    candidates += [ROOT/f'stage/docs/long_distance_20261101/{prefix}_T408.json' for prefix in ['release_receipt','source_manifest']]
    allowed={str(path.relative_to(ROOT)) for path in candidates if path.is_file()}
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())
    assert changed<=allowed,('Unrelated tracked edits; final sync deferred',sorted(changed-allowed))
    for name in sorted(allowed):
        path=ROOT/name
        assert path.suffix in ('.json','.md') and path.stat().st_size<2*1024**2
    if allowed:
        subprocess.run(['git','add','--',*sorted(allowed)],cwd=ROOT,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():
        subprocess.run(['git','commit','-m','Record bounded T408 joint-route results and verified evidence'],cwd=ROOT,check=True)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    return subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()

def driver():
    global ACTIVE
    p=plan();verify(p)
    start=time.time();deadline=start+p['all_seconds']
    install_handlers();signal.alarm(p['all_seconds']-1)
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline,external_timeout_seconds=p['all_seconds']))
    state='FAILED';stopped=[];source_state='NOT_ATTEMPTED'
    def call(file,args,seconds):
        global ACTIVE
        assert time.time()+5<deadline
        suffix='' if not args else '_'+args[0]
        with (DOC/('T408_'+file+suffix+'.log')).open('xb') as log:
            ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=ACTIVE.wait(timeout=min(seconds,max(1,deadline-time.time()-2)))
                assert code==0,(file,args,code)
            except BaseException:
                stop();raise
            finally:
                ACTIVE=None
    try:
        call('run.py',[],p['main_seconds']+10)
        stopped=native_stopped()
        call('audit.py',[],p['audit_seconds'])
        assert read(OUT/'audit.json')['state']=='PASS'
        stopped=native_stopped()
        state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        try:
            call('transport.py',['T408'],p['backup_seconds'])
            receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T408.json')
            assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['group']=='T408'
            state='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:
            save(DOC/'T408_BACKUP_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),training_restarted=False))
    except BaseException:
        stop()
        save(DOC/'T408_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
    finally:
        stop()
        try:
            stopped=native_stopped()
        except BaseException:
            state='FAILED_NATIVE_EXIT_NOT_VERIFIED'
        report=dict(state=state,elapsed_s=time.time()-start,finished_epoch=time.time(),result=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,all_native_pids_absent=stopped,all_owned_children_stopped=state!='FAILED_NATIVE_EXIT_NOT_VERIFIED',driver_pid=os.getpid(),driver_parent_pid=os.getppid(),narrow_paused=True,seven_eight_not_started=True,source_sync='See ignored run source_finalization.json and latest_source_sync_receipt.json; source is synchronized only after native exit')
        save(DOC/'T408_FINAL_REPORT.json',report)
        if state!='FAILED_NATIVE_EXIT_NOT_VERIFIED':
            try:
                target_tree=final_source_commit()
                call('transport.py',['source_sync'],p.get('source_seconds',120))
                receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
                assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==target_tree
                source_state='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED'
                save(OUT/'source_finalization.json',dict(state=source_state,root_tree=target_tree,remote_commit=receipt['remote_commit'],epoch=time.time()))
            except BaseException:
                source_state='PENDING_SOURCE_SYNC'
                save(OUT/'source_finalization.json',dict(state=source_state,error=traceback.format_exc(),epoch=time.time(),no_training_restart=True))
        signal.alarm(0)
        save(OUT/'driver_final_status.json',dict(state=state,source_state=source_state,updated_epoch=time.time(),pid=os.getpid(),deadline=deadline))
        print(json.dumps(dict(state=state,source_state=source_state)),flush=True)

def launch():
    p=plan();verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    with (DOC/'T408_CONSOLE.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-joint-wide','--why=Bounded joint LEFT RIGHT wide six-metre pilot','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'night.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launcher_pid=os.getpid(),epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),all_seconds=p['all_seconds'],external_timeout_seconds=p['all_seconds'],external_timeout_kill_after_seconds=2,outer_identity='systemd-inhibit -> timeout -> taskset -> run_stage.sh -> night.py driver'))
    print(json.dumps(dict(state='ACTUALLY_LAUNCHED',pid=proc.pid)),flush=True)

if __name__=='__main__':
    assert sys.argv[1:] in (['launch'],['driver'])
    globals()[sys.argv[1]]()
