"""Single diagnostic/finalizer. Source registration precedes a unique launch."""
from t411_core import *
import subprocess,signal
ACTIVE=None
def stop():
    global ACTIVE
    if ACTIVE and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM);ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
        except ProcessLookupError:pass
def driver():
    global ACTIVE
    p=plan();verify(p);start=time.time();deadline=min(start+p['all_seconds'],p['all_stop_absolute']);state='FAILED'
    def interrupted(*_):raise TimeoutError('T411 all-work deadline')
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(s,interrupted)
    signal.alarm(int(deadline-start)-2);save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline))
    def call(file,arguments,seconds):
        global ACTIVE
        suffix='' if not arguments else '_'+arguments[0]
        with (DOC/('T411_'+file+suffix+'.log')).open('xb') as f:
            ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*arguments],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:code=ACTIVE.wait(timeout=min(seconds,deadline-time.time()-3));assert code==0,(file,code)
            except BaseException:stop();raise
            finally:ACTIVE=None
    try:
        call('run.py',['main'],p['main_seconds']+2);s=read(OUT/'summary.json');assert not Path('/proc',str(s['pid'])).exists();call('audit.py',[],p['audit_seconds']);assert read(OUT/'audit.json')['state']=='PASS';state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        try:call('transport.py',['T411'],p['backup_seconds']);state='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:save(DOC/'T411_BACKUP_PENDING.json',dict(error=traceback.format_exc(),no_restart=True))
    except BaseException:stop();save(DOC/'T411_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_restart=True))
    finally:
        stop();signal.alarm(0);save(DOC/'T411_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),finished_epoch=time.time(),elapsed_s=time.time()-start,result=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,new_training=0,seven_eight=False));print(state,flush=True)
def launch():
    p=plan();verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==tree and r['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED';assert time.time()+p['all_seconds']<p['all_stop_absolute']
    with (DOC/'T411_CONSOLE.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-recovery-diagnostic','--why=Bounded actual-history closed-loop recovery','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=r['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),all_seconds=p['all_seconds']));print('T411_ACTUALLY_LAUNCHED',proc.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
