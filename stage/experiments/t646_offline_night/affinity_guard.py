"""Resource-only fallback when user systemd cpuset controller is unavailable.

Never edits scientific modules, protocols, raw arrays or model parameters.
Targets only threads already belonging to this exact Stage cgroup.
"""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/overnight_20261118'
CG=Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/stage-t646.service')
SAFE={8,10,12,14}

def main():
    os.sched_setaffinity(0,{14})
    # Membership makes all helper work count against the existing 400% CPU
    # quota, 4GiB RAM limit and absolute external cgroup kill deadline.
    (CG/'cgroup.procs').write_text(str(os.getpid()))
    p=json.loads((ROOT/'stage/runs/t646_offline_night/protocol.json').read_text())
    driver=json.loads((ROOT/'stage/runs/t646_offline_night/DRIVER_STARTED.json').read_text())['pid']
    start=time.time();changes=0
    receipt=dict(state='ACTIVE_RESOURCE_ONLY',pid=os.getpid(),driver=driver,cgroup=str(CG),safe_cpus=sorted(SAFE),
        reason='systemd EffectiveCPUs empty; Mesa disk helper thread widened its affinity',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_native=0,new_optimizer=0,epoch=start)
    (DOC/'AFFINITY_GUARD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    while time.time()<p['absolute_all_stop'] and Path('/proc',str(driver)).exists():
        for pid in (CG/'cgroup.procs').read_text().split():
            try:threads=list(Path('/proc',pid,'task').iterdir())
            except (FileNotFoundError,ProcessLookupError):continue
            for thread in threads:
                try:
                    tid=int(thread.name);mask=os.sched_getaffinity(tid)
                    if not mask<=SAFE:
                        os.sched_setaffinity(tid,mask&SAFE or {14});changes+=1
                except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(.5)
    receipt.update(state='ENDED',changes=changes,ended=time.time())
    (DOC/'AFFINITY_GUARD.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
