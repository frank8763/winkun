"""Wait for the approved disk reserve, then launch T412 at most once.

This administrative waiter performs no model import, optimization or physics.
It cannot lower the user's39GiB floor or extend any registered deadline.
"""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,subprocess,signal,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t412_closed_expert_likelihood'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);tmp=Path(str(p)+'.partial');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def used(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
def wait():
    p=read(OUT/'protocol.json');assert p['disk_exception']['user_approved_for']=='T412_ONLY';assert p['minimum_free_bytes']==39*2**30
    latest=min(p['learning_stop_absolute']-p['learning_seconds'],p['native_stop_absolute']-p['main_seconds']-5,p['all_stop_absolute']-p['all_seconds'])
    stable=0;state='WAITING_FOR_APPROVED_DISK_RESERVE'
    while time.time()<latest:
        if (OUT/'launch.json').exists():state='ACTUAL_T412_LAUNCH_EXISTS';break
        free=shutil.disk_usage(ROOT).free;own=used(OUT);needed=p['minimum_free_bytes']+max(0,p['final_bytes']-own)
        assert used(ROOT/'stage/runs')-own+p['final_bytes']<=2**30
        stable=stable+1 if free>=needed else 0
        save(OUT/'disk_admission.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),free_bytes=free,required_free_with_full_own_reserve=needed,minimum_free_bytes=p['minimum_free_bytes'],latest_launch_epoch=latest,stable_samples=stable,native=0,Adam=0,training_started=False))
        if stable>=3:
            assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
            for file,h in p['inputs'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
            assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
            with (OUT/'disk_admitted_launch.log').open('xb') as f:
                code=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py'),'launch'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
            state='ACTUALLY_LAUNCHED_T412' if code==0 and (OUT/'launch.json').exists() else 'PRELAUNCH_REJECTED_NO_AUTOMATIC_RETRY'
            break
        time.sleep(10)
    else:state='NOT_STARTED_NO_FULL_WINDOW_BEFORE_DAY_CUTOFF'
    save(OUT/'disk_admission.json',dict(state=state,pid=os.getpid(),updated_epoch=time.time(),native=0,Adam=0,training_started=(OUT/'launch.json').exists(),latest_launch_epoch=latest,actual_launch=read(OUT/'launch.json') if (OUT/'launch.json').exists() else None))
def launch():
    assert not (OUT/'disk_waiter_launch.json').exists() and not (OUT/'launch.json').exists()
    with (OUT/'disk_waiter_console.log').open('xb') as f:child=subprocess.Popen([sys.executable,str(__file__),'wait'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'disk_waiter_launch.json',dict(pid=child.pid,epoch=time.time(),native=0,Adam=0,mode='disk_admission_only_not_training',minimum_free_bytes=39*2**30));print('T412_DISK_ADMISSION_WAITER_STARTED',child.pid,flush=True)
if __name__=='__main__':
    assert sys.argv[1:] in (['wait'],['launch'])
    try:globals()[sys.argv[1]]()
    except BaseException:save(OUT/'disk_admission_error.json',dict(state='WAIT_OR_LAUNCH_STOPPED',error=traceback.format_exc(),native=0,Adam=0));raise
