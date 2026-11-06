"""Unique finite producer, audit, binary backup, then clean source publication."""
from t397_contract import *

def driver():
    p=read(OUT/'protocol.json');start=time.time();deadline=start+p['all_seconds'];child=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline))
    def call(file,mode,seconds,publish_status=True):
        nonlocal child
        # Source publication must not dirty an already committed tracked status.
        if publish_status:save(DOC/'T397_DRIVER_STATUS.json',dict(state=file+'_'+mode,pid=os.getpid(),updated_epoch=time.time()))
        with (DOC/('T397_'+file+'_'+mode+'.log')).open('wb') as log:
            child=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:code=child.wait(timeout=min(seconds,max(1,deadline-time.time()-1)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);time.sleep(.2)
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait();raise
            assert code==0,(file,mode,code)
            assert str(child.pid) not in subprocess.check_output(['ps','-eo','pgid='],text=True).split();child=None
    try:
        call('run.py','main',p['main_seconds']+10);call('run.py','audit',p['audit_seconds']);call('transport.py','T397',p['backup_seconds'])
        s=read(OUT/'summary.json');pre=read(OUT/'preflight.json');allcmd=s['commands']+pre['geometry']+pre['native_setup_commands'];assert allcmd<=p['max_commands']
        report=DOC/'T397_FINAL_REPORT.json';status=DOC/'T397_DRIVER_STATUS.json'
        save(report,dict(state='COMPLETE_DIAGNOSTIC_AUDIT_AND_RAW_BACKUP_PASS',actual=s,elapsed_before_source_s=time.time()-start,commands_including_preflight=allcmd,source_sync_final_evidence='stage/runs/t224_archive/latest_source_sync_receipt.json',receipt='stage/docs/long_distance_20261101/release_receipt_T397.json'))
        save(status,dict(state='COMPLETE_AUDIT_BACKUP_SOURCE_PUBLICATION_NEXT',pid=os.getpid(),updated_epoch=time.time()))
        files=[report,status,ROOT/'stage/docs/long_distance_20261101/release_receipt_T397.json',ROOT/'stage/docs/long_distance_20261101/source_manifest_T397.json']
        subprocess.run(['git','add',*[str(f.relative_to(ROOT)) for f in files]],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','commit','-m','Record bounded frozen T397 handoff diagnostic with exact audit and raw readback'],cwd=ROOT,check=True)
        call('transport.py','source_sync',p['source_seconds'],publish_status=False)
        print('T397_ALL_FINALIZERS_COMPLETE',flush=True)
    except BaseException:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);time.sleep(.2);os.killpg(child.pid,signal.SIGKILL);child.wait()
            except ProcessLookupError:pass
        save(DOC/'T397_DRIVER_ERROR.json',dict(error=traceback.format_exc(),elapsed_s=time.time()-start,pid=os.getpid()));raise

def launch():
    assert not (OUT/'launch.json').exists();p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS'
    rc=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert rc['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    with (DOC/'T397_driver.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--who=Stage-T397','--why=Bounded frozen two-duration handoff diagnostic','timeout','--signal=TERM','--kill-after=1',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=time.time(),source_remote_commit=rc['remote_commit'],root_tree=rc['root_tree'],all_seconds=p['all_seconds']));print(json.dumps(dict(state='LAUNCHED',pid=child.pid)))
if __name__=='__main__':globals()[sys.argv[1]]()
