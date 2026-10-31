"""Independent wall-clock watchdog. Never retries or launches another training."""
from pathlib import Path
import os,sys,subprocess,time,json,signal,datetime
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t255_overnight'
def stop_group(pid):
    for sig,pause in [(signal.SIGTERM,3),(signal.SIGKILL,1)]:
        try:os.killpg(pid,sig)
        except ProcessLookupError:return
        time.sleep(pause)
def run_bounded(script,deadline,name):
    with (OUT/(name+'.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (OUT/(name+'_process.json')).write_text(json.dumps(dict(pid=p.pid,process_group=p.pid,started_epoch=time.time(),deadline_epoch=deadline))+'\n')
        try:
            while p.poll() is None:
                if time.time()>=deadline:stop_group(p.pid);return dict(returncode=p.wait(),watchdog_timeout=True)
                time.sleep(2)
            return dict(returncode=p.returncode,watchdog_timeout=False)
        finally:
            # Workers of a failed interpreter may outlive the parent.
            stop_group(p.pid)
def main():
    assert not (OUT/'supervisor_started.json').exists(),'No duplicate supervisor'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.time();deadline=min(start+plan['max_wall_seconds']+60,plan['hard_deadline_epoch']-180)
    (OUT/'supervisor_started.json').write_text(json.dumps(dict(pid=os.getpid(),start_epoch=start,hard_deadline_epoch=deadline))+'\n')
    result=run_bounded('run.py',deadline,'pipeline')
    if not (OUT/'summary.json').exists():
        result['state']='WATCHDOG_OR_INTERPRETER_FAILURE';result['last_status']=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else None
        (OUT/'supervisor_failure.json').write_text(json.dumps(result,indent=2)+'\n')
        (ROOT/'stage/docs/overnight_20261031/MORNING_REPORT.md').write_text('# Stage 今夜未完整结束\n\n监督器已停止本轮进程组。不能声称5米通过或8米已运行。详情见t255_overnight/supervisor_failure.json及pipeline.log。\n')
    result['archive']=run_bounded('backup.py',min(time.time()+900,plan['hard_deadline_epoch']),'backup') if time.time()+60<plan['hard_deadline_epoch'] else dict(state='SKIPPED_NO_TIME')
    result['finished_epoch']=time.time();(OUT/'supervisor_summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
