"""Unique bounded producer/auditor/RAM-finalizer for T412.

The immutable source is synchronized before launch. Scientific execution never
waits for a network service; unavailable post-run transport remains PENDING.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,os,signal,subprocess,sys,time,traceback,shutil
import t412_learning as EXP

ROOT,HERE,OUT,DOC=EXP.ROOT,EXP.HERE,EXP.OUT,EXP.DOC
read,save,sha,N=EXP.read,EXP.save,EXP.sha,EXP.N
ACTIVE=None

def plan():
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    return read(OUT/'protocol.json')

def absolute_deadlines(p):
    tz=ZoneInfo('Asia/Shanghai')
    fixed={key:datetime(2026,11,7,hour,minute,tzinfo=tz).timestamp()
           for key,hour,minute in [('learning',18,45),('native',19,30),('all',20,0)]}
    result=dict(learning=min(fixed['learning'],p.get('learning_stop_absolute',p.get('learning_stop_epoch',fixed['learning']))),
                native=min(fixed['native'],p.get('native_stop_absolute',p.get('native_stop_epoch',fixed['native']))),
                all=min(fixed['all'],p.get('all_stop_absolute',p.get('all_stop_epoch',fixed['all']))))
    assert result['learning']<result['native']<result['all']
    return result

def verify(p):
    for path,digest in p.get('inputs',p.get('input_hashes',{})).items():
        assert sha(path)==digest,('Registered input changed',path)
    N.science(p)
    own_limit=p.get('final_bytes',20*2**20)
    assert own_limit<=20*2**20
    N.capacity(OUT,own_limit)
    used=lambda directory:sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())
    assert used(ROOT/'stage/runs')-used(OUT)+own_limit<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30

def native_stopped():
    pids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        path=OUT/name
        if not path.exists():continue
        value=read(path)
        pids.update(value[key] for key in ['pid','parent'] if type(value.get(key)) is int)
        pids.update(value.get('workers',[]))
    for pid in pids:
        assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
    return sorted(pids)

def stop():
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:os.killpg(ACTIVE.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:ACTIVE.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(ACTIVE.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            ACTIVE.wait(timeout=2)

def install_handlers():
    def interrupted(signum,frame):raise TimeoutError('T412 driver stop '+str(signum))
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(signum,interrupted)

def driver():
    global ACTIVE
    p=plan();verify(p);absolute=absolute_deadlines(p);launch=read(OUT/'launch.json')
    assert not (OUT/'driver_started.json').exists()
    start=time.time();deadline=min(start+p.get('all_seconds',7200),launch['epoch']+p.get('all_seconds',7200),absolute['all'])
    assert start+10<deadline
    install_handlers();signal.alarm(max(1,int(deadline-start-5)))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline,
          absolute_learning_stop=absolute['learning'],absolute_native_stop=absolute['native'],absolute_all_stop=absolute['all']))
    state='FAILED';absent=[]
    def call(file,args,seconds,cutoff):
        global ACTIVE
        remaining=min(seconds,cutoff-time.time()-3,deadline-time.time()-3)
        assert remaining>0,('No registered child wall reserve',file)
        suffix='' if not args else '_'+args[0]
        with (DOC/('T412_'+file+suffix+'.log')).open('xb') as log:
            ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=ACTIVE.wait(timeout=remaining)
                assert code==0,(file,args,code)
            except BaseException:stop();raise
            finally:ACTIVE=None
    try:
        call('run.py',[],p.get('main_seconds',5400)+2,absolute['native'])
        summary=read(OUT/'summary.json');assert summary['state']=='COMPLETE'
        absent=native_stopped()
        call('audit.py',[],p.get('audit_seconds',900),deadline)
        assert read(OUT/'audit.json')['state']=='PASS';absent=native_stopped()
        state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        try:
            receipt_path=ROOT/'stage/docs/long_distance_20261101/release_receipt_T412.json'
            if receipt_path.exists():
                receipt=read(receipt_path)
                assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['group']=='T412'
            else:
                call('transport.py',['T412'],p.get('backup_seconds',300),deadline)
                receipt=read(receipt_path)
                assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['group']=='T412'
            state='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:
            save(DOC/'T412_BACKUP_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time(),no_training_restart=True))
    except BaseException:
        stop();save(DOC/'T412_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),restart=False))
    finally:
        stop()
        try:absent=native_stopped();all_native_stopped=True
        except BaseException:all_native_stopped=False;state='FAILED_NATIVE_EXIT_NOT_VERIFIED'
        signal.alarm(0)
        save(DOC/'T412_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),parent=os.getppid(),finished_epoch=time.time(),
             elapsed_s=time.time()-start,result=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,
             all_native_pids_absent=absent,all_owned_children_stopped=all_native_stopped,
             source_sync='Registered source verified before launch; post-run outcome/source sync is root-managed, not automatic',
             seven_eight_entered=False,deadline=deadline))
        print(json.dumps(dict(state=state,all_native_stopped=all_native_stopped)),flush=True)

def launch():
    p=plan();verify(p);absolute=absolute_deadlines(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();all_seconds=p.get('all_seconds',7200)
    assert now+p.get('learning_seconds',3600)<absolute['learning']
    assert now+p.get('main_seconds',5400)+2<absolute['native']
    assert now+all_seconds<absolute['all']
    with (DOC/'T412_CONSOLE.log').open('xb') as log:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T412',
             '--why=Bounded closed-expert likelihood learning until20:00','timeout','--signal=TERM','--kill-after=2',
             str(all_seconds),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'driver'],
             cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,launcher_pid=os.getpid(),epoch=now,source_remote_commit=receipt['remote_commit'],root_tree=tree,
                protocol_sha256=sha(OUT/'protocol.json'),all_seconds=all_seconds,
                learning_deadline=min(now+p.get('learning_seconds',3600),absolute['learning']),
                native_deadline=min(now+p.get('main_seconds',5400),absolute['native']),
                all_deadline=min(now+all_seconds,absolute['all']),hard_all_stop_Beijing='2026-11-07T20:00:00+08:00')
    save(OUT/'launch.json',record)
    print(json.dumps(dict(state='ACTUALLY_LAUNCHED_T412',pid=process.pid,all_deadline=record['all_deadline'])),flush=True)

if __name__=='__main__':
    assert sys.argv[1:] in (['launch'],['driver'])
    globals()[sys.argv[1]]()
