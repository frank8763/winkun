"""Unique offline finalizer; scientific completion never waits for internet."""
from night import *
ACTIVE=None

def stop_active():
    global ACTIVE
    if ACTIVE and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait()

def call(name,seconds):
    global ACTIVE
    p=load_plan();remaining=min(seconds,p['all_stop_epoch']-time.time()-20);assert remaining>0
    with (OUT/(name+'.log')).open('xb') as f:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/name)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=ACTIVE.wait(timeout=remaining);assert code==0,(name,code)
        except BaseException:stop_active();raise
        finally:ACTIVE=None

def main():
    N.offline();p=load_plan();verify(p);assert not (OUT/'driver_started.json').exists();start=time.time()
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,absolute_stop=p['all_stop_epoch'],network_required=False))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T402 all-work stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T40208:30absolute stop')));signal.alarm(max(1,int(p['all_stop_epoch']-time.time())))
    final='STOPPED_WITH_ERROR';result={}
    try:
        call('run.py',min(p['main_seconds']+10,p['native_stop_epoch']-time.time()+5))
        s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and s['all_workers_stopped'] and s['all_native_sessions_closed']
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        call('audit.py',p['audit_seconds']);assert read(OUT/'audit.json')['state']=='PASS'
        final='COMPLETE_FINITE_LOCAL_CHAIN_AND_AUDIT_REMOTE_PENDING';result=s
    except BaseException:
        stop_active();save(OUT/'driver_error.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
        if (OUT/'summary.json').exists():result=read(OUT/'summary.json')
    finally:
        stop_active();signal.alarm(0)
        report=dict(state=final,finished_epoch=time.time(),elapsed_s=time.time()-start,result=result,network_required=False,remote_backup_pending=True,source_sync_pending=True,all_owned_children_stopped=True,all_stop_epoch=p['all_stop_epoch'],global_runs_bytes=N.K.S.used(ROOT/'stage/runs'),free_bytes=shutil.disk_usage(ROOT).free,entered7or8=False)
        save(OUT/'final_report.json',report);save(DOC/'FINAL_REPORT.json',report)
        (DOC/'FINAL_REPORT.md').write_text('# Stage offline night actual result\n\nState: '+final+'\n\nAll scientific steps are finite and preprogrammed. No network or LLM is required. Remote backup remains PENDING until full binary/member readback after reconnection. Full six-metre publication gate and seven/eight metres are not claimed.\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        save(DOC/'NIGHT_STATUS.json',dict(state=final,updated_epoch=time.time(),driver_pid=os.getpid(),all_owned_children_stopped=True,remote_pending=True));N.manifest(OUT)
        print(json.dumps(dict(state=final,finished_epoch=time.time(),remote_pending=True)),flush=True)

def launch():
    p=load_plan();verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not any((OUT/f).exists() for f in ['launch.json','driver_started.json','summary.json'])
    assert time.time()+3600<p['learning_stop_epoch'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==tree
    with (DOC/'NIGHT_CONSOLE.log').open('xb') as f:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-offline-T402','--why=Preprogrammed Stage offline night ends08:30','timeout','--signal=TERM','--kill-after=2',str(max(1,int(p['all_stop_epoch']-time.time()))),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'main'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    launch=dict(pid=child.pid,epoch=time.time(),source_local_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_remote_commit=receipt['remote_commit'],source_root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),absolute_stop_epoch=p['all_stop_epoch'],offline=True,network_required=False)
    save(OUT/'launch.json',launch);save(DOC/'NIGHT_LAUNCH.json',launch);print(json.dumps(dict(state='ACTUALLY_LAUNCHED_OFFLINE_BACKGROUND',pid=child.pid,absolute_stop=p['all_stop_epoch'])))
if __name__=='__main__':globals()[sys.argv[1]]()
