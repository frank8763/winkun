"""One bounded last attempt; conditional wide admission after independent audit."""
from t404_core import *
ACTIVE=None
def stop():
    global ACTIVE
    if ACTIVE and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM);ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
        except ProcessLookupError:pass
def driver():
    global ACTIVE
    p=plan();verify(p);start=time.time();deadline=start+p['all_seconds'];save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline));state='FAILED';wide=False
    def call(file,args,seconds):
        global ACTIVE
        assert time.time()+5<deadline
        with (DOC/('T404_'+file+'_'+str(len(list(DOC.glob('T404_*.log'))))+'.log')).open('xb') as log:
            ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=ACTIVE.wait(timeout=min(seconds,deadline-time.time()-2));assert code==0,(file,code)
            except BaseException:stop();raise
            finally:ACTIVE=None
    try:
        call('run.py',[],p['main_seconds']+10);call('audit.py',[],p['audit_seconds']);s=read(OUT/'summary.json');assert read(OUT/'audit.json')['state']=='PASS'
        if not s['result']['stable_six']:
            call('wide.py',['main'],900);call('wide.py',['audit'],180);wide=True
        state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        for group in ['T404']+(['T405'] if wide else []):
            try:call('transport.py',[group],p['backup_seconds'])
            except BaseException:save(DOC/f'{group}_BACKUP_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),training_restarted=False))
        if all((ROOT/f'stage/docs/long_distance_20261101/release_receipt_{g}.json').exists() for g in ['T404']+(['T405'] if wide else [])):state='COMPLETE_AUDITED_REMOTE_VERIFIED'
    except BaseException:stop();save(DOC/'T404_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
    finally:
        stop();report=dict(state=state,finished_epoch=time.time(),elapsed_s=time.time()-start,narrow_result=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,wide_started=wide,wide_result=read(ROOT/'stage/runs/t405_wide_admission/summary.json') if (ROOT/'stage/runs/t405_wide_admission/summary.json').exists() else None,next_action='Use separate wide6→7→8 protocols if narrow gate failed; no extra narrow cycles',all_owned_children_stopped=True)
        save(DOC/'T404_FINAL_REPORT.json',report);save(DOC/'T404_DRIVER_STATUS.json',dict(state=state,updated_epoch=time.time(),pid=os.getpid()));print(json.dumps(dict(state=state,wide_started=wide)),flush=True)
def launch():
    p=plan();verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['root_tree']==tree
    with (DOC/'T404_CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-last-narrow','--why=Final bounded narrow attempt then wide admission','timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),all_seconds=p['all_seconds']));print(json.dumps(dict(state='ACTUALLY_LAUNCHED',pid=child.pid)))
if __name__=='__main__':globals()[sys.argv[1]]()
