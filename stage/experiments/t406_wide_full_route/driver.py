"""Unique bounded wide-route producer/auditor/evidence finalizer."""
from t406_core import *
ACTIVE=None
def install_handlers():
    def interrupted(signum,frame):raise TimeoutError('T406 driver interrupted '+str(signum))
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(signum,interrupted)
    return interrupted
def stop():
    global ACTIVE
    if ACTIVE and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM);ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
        except ProcessLookupError:pass
def driver():
    global ACTIVE
    p=plan();verify(p);start=time.time();deadline=start+p['all_seconds'];install_handlers();signal.alarm(p['all_seconds']-1);save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline));status='FAILED'
    def call(file,args,seconds):
        global ACTIVE
        with (DOC/('T406_'+file+'.log')).open('xb') as log:
            ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=ACTIVE.wait(timeout=min(seconds,max(1,deadline-time.time()-2)));assert code==0,(file,code)
            except BaseException:stop();raise
            finally:ACTIVE=None
    try:
        call('run.py',[],p['main_seconds']+10);call('audit.py',[],p['audit_seconds']);status='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        try:call('transport.py',['T406'],p['backup_seconds']);status='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:save(DOC/'T406_BACKUP_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
    except BaseException:stop();save(DOC/'T406_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
    finally:
        stop();signal.alarm(0);save(DOC/'T406_FINAL_REPORT.json',dict(state=status,elapsed_s=time.time()-start,finished_epoch=time.time(),result=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,all_owned_children_stopped=True,narrow_paused=True,seven_eight_not_started=True));print(json.dumps(dict(state=status)),flush=True)
def launch():
    p=plan();verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    with (DOC/'T406_CONSOLE.log').open('xb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-wide-route','--why=Bounded wide six-metre full-route pilot','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),all_seconds=p['all_seconds']));print(json.dumps(dict(state='ACTUALLY_LAUNCHED',pid=proc.pid)))
if __name__=='__main__':globals()[sys.argv[1]]()
