"""Unique bounded producer/auditor/remote finalizer, with actual process IDs."""
from t392_learning import *

def driver():
    p=read(OUT/'protocol.json');start=time.time();deadline=start+p['all_seconds'];child=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline))
    def call(file,mode,seconds):
        nonlocal child
        save(DOC/'T392_DRIVER_STATUS.json',dict(state=file+'_'+mode,pid=os.getpid(),updated_epoch=time.time()))
        with (DOC/('T392_'+file+'_'+mode+'.log')).open('wb') as log:
            child=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:code=child.wait(timeout=min(seconds,max(1,deadline-time.time()-1)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);time.sleep(.2)
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait();raise
            assert code==0,(file,mode,code)
            assert str(child.pid) not in subprocess.check_output(['ps','-eo','pgid='],text=True).split()
            child=None
    try:
        call('run.py','main',p['main_seconds']+10)
        call('run.py','audit',p['audit_seconds'])
        call('transport.py','T392',p['backup_seconds'])
        summary=read(OUT/'summary.json')
        all_commands=summary['commands']+read(OUT/'preflight.json')['geometry']
        assert all_commands<=p['max_commands'] and summary['physics']<=p['max_physics']
        save(DOC/'T392_FINAL_REPORT.json',dict(state='COMPLETE_AUDITED_REMOTE_VERIFIED',actual=summary,elapsed_s=time.time()-start,
            commands_including_zero_native_geometry=all_commands,receipt=str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T392.json')))
        save(DOC/'T392_DRIVER_STATUS.json',dict(state='COMPLETE',pid=os.getpid(),updated_epoch=time.time()))
    except BaseException:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);time.sleep(.2);os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass
        save(DOC/'T392_DRIVER_ERROR.json',dict(error=traceback.format_exc(),elapsed_s=time.time()-start,pid=os.getpid()));raise

def launch():
    assert not (OUT/'launch.json').exists();p=read(OUT/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS'
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    with (DOC/'T392_driver.log').open('wb') as log:
        proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--who=Stage-T392','--why=Bounded new070 demo-assisted closed-loop learning',
            'timeout','--signal=TERM','--kill-after=1',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],all_seconds=p['all_seconds']))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))

if __name__=='__main__':globals()[sys.argv[1]]()
