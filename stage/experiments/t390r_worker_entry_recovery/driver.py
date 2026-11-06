from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t390r_worker_entry_recovery';DOC=ROOT/'stage/docs/review_20261106'
def read(p):return json.loads(Path(p).read_text())
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def driver():
    p=read(OUT/'protocol.json');start=time.time();current=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,all_deadline=p['absolute_all_deadline']))
    def stop(*_):raise TimeoutError('T390 finalizer absolute stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(p['absolute_all_deadline']-start)-2))
    try:
        for name,script,mode,seconds in [('native','run.py','main',p['main_seconds']),('audit','run.py','audit',p['audit_seconds']),('backup','transport.py','T389-T390R',p['backup_seconds'])]:
            save(DOC/'T390R_DRIVER_STATUS.json',dict(state=name,pid=os.getpid(),updated_epoch=time.time()))
            with (DOC/('T390R_'+name+'.log')).open('wb') as log:
                current=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=current.wait(timeout=min(seconds,max(1,int(p['absolute_all_deadline']-time.time())-1)))
                except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGTERM);time.sleep(.2);os.killpg(current.pid,signal.SIGKILL);current.wait();raise
                assert code==0,(name,code)
                current=None
        s=read(OUT/'summary.json');save(DOC/'T390R_FINAL_REPORT.json',dict(state='COMPLETE_AUDIT_AND_REMOTE_VERIFIED',elapsed_s=time.time()-start,
            results=s['results'],physics=s['physics'],commands=s['commands'],new_RL_updates=0,
            backup_receipt=str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T389-T390R.json')))
        save(DOC/'T390R_DRIVER_STATUS.json',dict(state='COMPLETE',pid=os.getpid(),updated_epoch=time.time()))
    except BaseException:
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM);time.sleep(.2);os.killpg(current.pid,signal.SIGKILL);current.wait()
            except ProcessLookupError:pass
        save(DOC/'T390R_DRIVER_ERROR.json',dict(error=traceback.format_exc(),pid=os.getpid(),elapsed_s=time.time()-start));raise
    finally:signal.alarm(0)
def launch():
    assert not (OUT/'launch.json').exists()
    p=read(OUT/'protocol.json');assert read(OUT/'preflight.json')['state']=='PASS'
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    cmd=['systemd-inhibit','--what=sleep','--mode=block','--who=Stage-T390R','--why=Bounded single-Actor experiment','timeout','--signal=TERM','--kill-after=1',str(max(1,int(p['absolute_all_deadline']-time.time()))),
         'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver']
    with (DOC/'T390R_driver.log').open('wb') as log:proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],all_deadline=p['absolute_all_deadline']))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))
if __name__=='__main__':globals()[sys.argv[1]]()
