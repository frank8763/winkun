"""Strict Phase resource gate; pause only this Stage computation group."""
from pathlib import Path
import json,os,signal,sys,time
from status_io import read_status
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t665_remaining_verification'
CG_SUFFIX='/stage-t665.service'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import control as Raw

def is_overlap(error):
    value=error.args[0] if error.args else None
    return isinstance(value,tuple) and value and value[0]=='Phase physical CPU overlap; do not launch'

def wait_for_clear(probe,deadline,pause,sleep=time.sleep,clock=time.time):
    waits=0
    while clock()<deadline:
        try:return probe(),waits
        except AssertionError as error:
            if not is_overlap(error):raise
            waits+=1;pause();sleep(.25)
    raise TimeoutError('Stage resource wait reached original bounded deadline')

def signal_stage_group(pid,sig,cgroup_suffix=CG_SUFFIX):
    try:
        assert pid>1 and os.getpgid(pid)==pid,'Refuse to signal a shared process group'
        assert cgroup_suffix in Path('/proc',str(pid),'cgroup').read_text(),'Refuse to signal outside this Stage cgroup'
        os.killpg(pid,sig);return True
    except (FileNotFoundError,ProcessLookupError):return False

def resources():
    # Outside registered execution this remains the original strict admission
    # check. No wait is permitted to turn a failed preflight into a pass.
    f=OUT/'protocol.json'
    if not f.exists():return Raw.resources()
    p=json.loads(f.read_text())
    def pause():
        sf=OUT/'status.json';status=read_status(sf)
        if status.get('child_pid')==os.getpid():
            # Runtime/audit has its own session. Stop includes its workers;
            # helper stays runnable to resume after a fresh strict probe.
            signal_stage_group(os.getpid(),signal.SIGSTOP)
    result,waits=wait_for_clear(Raw.resources,p['absolute_all_stop']-p['finalization_seconds'],pause)
    return dict(result,resource_wait_probe_count=waits)

def helper_gate(status,probe=Raw.resources):
    pid=status.get('child_pid')
    try:probe();clear=True;reason=None
    except AssertionError as error:
        if not is_overlap(error):raise
        clear=False;reason=str(error)
    if pid and Path('/proc',str(pid)).exists():
        signal_stage_group(pid,signal.SIGCONT if clear else signal.SIGSTOP)
    return clear,reason,pid
