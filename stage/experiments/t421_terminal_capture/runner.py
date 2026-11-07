"""Unique bounded launcher/finalizer for the terminal-capture teacher pilot.

The user's new independent request after21:00 does not resume the old evening
chain. Scientific work ends within its own relative and absolute limits; this
driver never starts Actor training or a longer-distance navigation task.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,os,signal,subprocess,sys,time,traceback,shutil

import science as EXP

ROOT,HERE,OUT,DOC=EXP.ROOT,EXP.HERE,EXP.OUT,EXP.DOC
read,save,sha=EXP.read,EXP.save,EXP.sha
ACTIVE=None

def absolute_stop(p):
    fixed=datetime(2026,11,7,22,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    return min(fixed,p.get('all_stop_absolute',p.get('absolute_stop',fixed)))

def plan():
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    return read(OUT/'protocol.json')

def verify(p):
    EXP.verify(p)
    assert p['all_seconds']<=1800
    assert p['main_seconds']<=900 and p['audit_seconds']<=180 and p['backup_seconds']<=180
    assert EXP.E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30

def native_stopped(require_complete=True):
    summary=read(OUT/'summary.json')
    if require_complete:
        assert summary['state']=='COMPLETE'
        assert summary['all_sessions_closed'] is True
    assert summary.get('Actor_updates',0)==summary.get('optimizer_steps',0)==0
    ids=set()
    for name in ['main_started.json','summary.json']:
        f=OUT/name
        if f.exists():
            value=read(f)
            ids.update(value[k] for k in ['pid','main_pid'] if type(value.get(k)) is int)
    assert ids
    for pid in ids:assert not Path('/proc',str(pid)).exists(),('Live or reused native PID',pid)
    return sorted(ids)

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

def call(file,args,seconds,deadline,log):
    global ACTIVE
    remaining=min(seconds,deadline-time.time()-5)
    assert remaining>0,('No complete child execution window',file)
    with Path(log).open('xb') as stream:
        ACTIVE=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/file),*args],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=ACTIVE.wait(timeout=remaining)
            assert code==0,(file,args,code)
        except BaseException:stop();raise
        finally:ACTIVE=None

def launch():
    p=plan();verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS'
    assert not (OUT/'launch.json').exists() and not (OUT/'main_started.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();deadline=min(now+p['all_seconds'],absolute_stop(p))
    assert now+p['all_seconds']<absolute_stop(p)
    assert p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p.get('source_seconds',120)+30<=p['all_seconds']
    with (DOC/'T421_CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T421',
            '--why=Independent bounded terminal-capture teacher verification','timeout','--signal=TERM','--kill-after=2',
            str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'runner.py'),'driver'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,launcher_pid=os.getpid(),epoch=now,deadline=deadline,
        relative_all_seconds=p['all_seconds'],absolute_stop=absolute_stop(p),
        source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),
        Actor_updates=0,optimizer_steps=0,previous_evening_chain_resumed=False,
        authorization='User new independent terminal-capture start request after21:00,2026-11-07',
        no_automatic_Actor_training=True,seven_eight_entered=False))
    print(json.dumps(dict(state='T421_ACTUALLY_LAUNCHED',pid=child.pid,deadline=deadline)),flush=True)

def driver():
    p=plan();verify(p);launch=read(OUT/'launch.json')
    assert not (OUT/'driver_started.json').exists()
    start=time.time();deadline=min(launch['deadline'],launch['epoch']+p['all_seconds'],absolute_stop(p))
    assert start+30<deadline
    save(OUT/'driver_started.json',dict(pid=os.getpid(),parent=os.getppid(),epoch=start,deadline=deadline,
        Actor_updates=0,optimizer_steps=0,no_previous_run_restart=True))
    def interrupted(signum,frame):raise TimeoutError('T421 hard deadline or interruption '+str(signum))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    signal.alarm(max(1,int(deadline-time.time()-3)))
    state='FAILED';absent=[];error=None;backup_error=None
    execution_cutoff=deadline-p.get('source_seconds',120)
    try:
        call('science.py',['main'],p['main_seconds'],execution_cutoff,DOC/'T421_main.log')
        absent=native_stopped()
        call('audit.py',[],p['audit_seconds'],execution_cutoff,DOC/'T421_audit.log')
        assert read(OUT/'audit.json')['state']=='PASS';absent=native_stopped()
        state='COMPLETE_AUDITED_LOCAL_REMOTE_PENDING'
        receipt_path=ROOT/'stage/docs/long_distance_20261101/release_receipt_T421.json'
        try:
            if not receipt_path.exists():call('transport.py',['T421'],p['backup_seconds'],execution_cutoff,DOC/'T421_backup.log')
            receipt=read(receipt_path)
            assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['group']=='T421'
            state='COMPLETE_AUDITED_REMOTE_VERIFIED'
        except BaseException:
            backup_error=traceback.format_exc()
            save(DOC/'T421_BACKUP_PENDING.json',dict(error=backup_error,epoch=time.time(),no_science_restart=True))
    except BaseException:
        error=traceback.format_exc();stop()
        save(DOC/'T421_DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_training_or_native_restart=True))
    finally:
        stop();signal.alarm(0)
        try:
            if (OUT/'summary.json').exists():absent=native_stopped(require_complete=False)
            native_absent=bool(absent)
        except BaseException:native_absent=False
        save(DOC/'T421_FINAL_REPORT.json',dict(state=state,pid=os.getpid(),parent=os.getppid(),finished_epoch=time.time(),
            elapsed_s=time.time()-start,deadline=deadline,all_native_pids_absent=absent,all_owned_native_processes_stopped=native_absent,
            Actor_updates=0,optimizer_steps=0,no_automatic_Actor_training=True,seven_eight_entered=False,
            error=error,backup_error=backup_error,
            source_sync='Root-managed outcome commit and exact root-tree sync remain separate; no automatic source mutation by this driver.'))
        print(json.dumps(dict(state=state,all_native_stopped=native_absent,Actor_updates=0,optimizer_steps=0)),flush=True)

if __name__=='__main__':
    assert sys.argv[1:] in [['launch'],['driver']]
    globals()[sys.argv[1]]()
