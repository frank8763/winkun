"""Resource-only fallback when user systemd cpuset controller is unavailable.

Never edits scientific modules, protocols, raw arrays or model parameters.
Targets only threads already belonging to this exact Stage cgroup.
"""
from pathlib import Path
import hashlib,json,os,sys,time
import t689_wait as Wait
from status_io import read_status
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t689_reference_functional_geometry'
DOC=ROOT/'stage/docs/weekend_20261120_23/t689'
CG=Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/stage-t689.service')
SAFE={8,10,12,14}

def configuration(folder=OUT):
    p=json.loads((folder/'protocol.json').read_text())
    assert p['out']==str(OUT)
    return p,folder/'DRIVER_STARTED.json',folder/'AFFINITY_STOP'

def main():
    p,driver_file,stop_file=configuration()
    driver=json.loads(driver_file.read_text())['pid']
    os.sched_setaffinity(0,{14})
    # Membership makes all helper work count against the existing 400% CPU
    # quota, 4GiB RAM limit and absolute external cgroup kill deadline.
    (CG/'cgroup.procs').write_text(str(os.getpid()))
    start=time.time();changes=0;pauses=0;was_clear=True
    receipt=dict(state='ACTIVE_RESOURCE_ONLY',pid=os.getpid(),driver=driver,cgroup=str(CG),safe_cpus=sorted(SAFE),
        reason='systemd EffectiveCPUs empty; Mesa disk helper thread widened its affinity',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_native=0,new_optimizer=0,epoch=start)
    (DOC/'AFFINITY_GUARD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    while time.time()<p['absolute_all_stop'] and Path('/proc',str(driver)).exists() and not stop_file.exists():
        sf=OUT/'status.json'
        status=read_status(sf)
        clear,reason,child_pid=Wait.helper_gate(status)
        if clear!=was_clear:
            if not clear:pauses+=1
            value=dict(state='RESOURCE_CLEAR' if clear else 'STAGE_COMPUTE_PAUSED_FOR_PHASE_CPU',epoch=time.time(),child_pid=child_pid,reason=reason,pauses=pauses)
            (DOC/'RESOURCE_WAIT.json').write_text(json.dumps(value,indent=2)+'\n')
            was_clear=clear
        for pid in (CG/'cgroup.procs').read_text().split():
            try:threads=list(Path('/proc',pid,'task').iterdir())
            except (FileNotFoundError,ProcessLookupError):continue
            for thread in threads:
                try:
                    tid=int(thread.name);mask=os.sched_getaffinity(tid)
                    if not mask<=SAFE:
                        os.sched_setaffinity(tid,mask&SAFE or {14});changes+=1
                except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(.25)
    receipt.update(state='ENDED',changes=changes,resource_pauses=pauses,ended=time.time())
    (DOC/'AFFINITY_GUARD.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
