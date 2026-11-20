"""Reproduce captured strict-probe abort; verify owned-group pause/resume."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
import t664r_wait as W
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261120/t664r'

class Cmd:
    def read_bytes(self):return b'python\0/home/linghu/Documents/Codex/2026-10-31/new-chat/phase-/resource-fixture\0'
class Task:
    name='99999999'
class Tasks:
    def iterdir(self):return [Task()]
class Proc:
    name='377408'
    def __truediv__(self,key):return Cmd() if key=='cmdline' else Tasks()
class Procs:
    def iterdir(self):return [Proc()]

def captured_overlap():
    oldpath,oldaff=W.Raw.Path,os.sched_getaffinity
    W.Raw.Path=lambda *args:Procs() if args==('/proc',) else Path(*args)
    os.sched_getaffinity=lambda pid:{8,10,12,14,24,26,28,30} if pid==99999999 else oldaff(pid)
    try:return W.Raw.resources()
    finally:W.Raw.Path=oldpath;os.sched_getaffinity=oldaff

def main():
    if sys.argv[1:] == ['legacy']:
        captured_overlap();raise AssertionError('Captured CPU overlap not detected')
    # Actual legacy code reaches the captured physical-core assertion.
    try:captured_overlap()
    except AssertionError as error:assert W.is_overlap(error);legacy=str(error)
    else:raise AssertionError('Legacy signal not reproduced')
    pause_calls=[];sleep_calls=[];counter=[0]
    def transient():
        counter[0]+=1
        return captured_overlap() if counter[0]<3 else dict(state='PASS')
    result,waits=W.wait_for_clear(transient,10,lambda:pause_calls.append(True),sleep=lambda s:sleep_calls.append(s),clock=lambda:0)
    assert result['state']=='PASS' and waits==2 and len(pause_calls)==2 and len(sleep_calls)==2
    try:W.wait_for_clear(lambda:(_ for _ in ()).throw(AssertionError('Not a resource overlap')),10,lambda:None,clock=lambda:0)
    except AssertionError as error:assert str(error)=='Not a resource overlap'
    else:raise AssertionError('Unrelated contract swallowed')
    try:W.wait_for_clear(captured_overlap,0,lambda:None,clock=lambda:0)
    except TimeoutError:pass
    else:raise AssertionError('Expired deadline extended')
    # Exercise SIGSTOP/SIGCONT on a newly owned inert process group, never
    # touching any Phase process or creating native/optimizer work.
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    try:
        cg=Path('/proc',str(child.pid),'cgroup').read_text().split(':',2)[-1].strip()
        assert W.signal_stage_group(child.pid,signal.SIGSTOP,cg)
        deadline=time.time()+2
        while time.time()<deadline and '\nState:\tT' not in Path('/proc',str(child.pid),'status').read_text():time.sleep(.01)
        assert '\nState:\tT' in Path('/proc',str(child.pid),'status').read_text()
        assert W.signal_stage_group(child.pid,signal.SIGCONT,cg)
        deadline=time.time()+2
        while time.time()<deadline and '\nState:\tT' in Path('/proc',str(child.pid),'status').read_text():time.sleep(.01)
        assert '\nState:\tT' not in Path('/proc',str(child.pid),'status').read_text()
        try:W.signal_stage_group(child.pid,signal.SIGSTOP,'/not-this-Stage-cgroup')
        except AssertionError:pass
        else:raise AssertionError('Foreign cgroup signal permitted')
    finally:
        os.killpg(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=3)
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',legacy_actual_scanner_reproduced=legacy,
        transient_overlap_paused_before_clear=True,strict_overlap_probe_preserved=True,
        actual_owned_inert_group_STOP_CONT=True,foreign_cgroup_refused=True,
        unrelated_contract_still_raises=True,original_deadline_not_extended=True,
        Phase_resource_metadata_only=True,new_native=0,new_optimizer=0)
    (DOC/'RESOURCE_REGRESSION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
