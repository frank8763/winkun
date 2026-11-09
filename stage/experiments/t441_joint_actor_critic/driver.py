"""Single finalizer; no network needed for scientific work or local audit."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t441_joint_actor_critic';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    proc=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        out,_=proc.communicate(timeout=max(1,seconds))
        if proc.returncode:raise RuntimeError(f'{script} exit {proc.returncode}: '+out.decode(errors='replace')[-6000:])
        return out.decode(errors='replace')
    except BaseException:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        raise

def main():
    p=read(OUT/'protocol.json');launch=read(OUT/'launch.json');end=launch['all_stop_epoch']
    def stop(*_):raise TimeoutError('T441 external supervisor stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop_epoch=end))
    report=None
    try:
        child('run.py','main',seconds=min(p['main_seconds']+20,end-time.time()-2000))
        child('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-850))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
        ids=[s['pid'],a['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',physics=s['physics'],commands=s['commands']+16,
            main_seconds=s['elapsed_s'],rounds_completed=len(s['rounds']),live_Actor_Adam=s['live_Actor_Adam'],live_V_Adam=s['live_V_Adam'],
            shadow_Actor_Adam=s['live_Actor_Adam'],shadow_V_Adam=s['live_V_Adam'],
            audit_discarded_historical_Adam=a['discarded_historical_Actor_plus_V_Adam'],
            development=s['development'],fresh16=None,reload78=None,six_gate=False,eight_started=False,
            source='T427 research warm-start, not safe deployment promotion',remote='PENDING',
            all_native_and_audit_processes_stopped=True,finished_epoch=time.time())
        save(DOC/'T441_FINAL_REPORT.json',report)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        # No writes/prints inside OUT after sealing, including driver console.
        try:
            child('transport.py','T441',seconds=min(650,end-time.time()-220))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T441.json')['state']
        except BaseException:
            save(DOC/'T441_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T441_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text(f"2026-11-09 T441联合循环Actor–Critic集成已结束，4轮目标/实际{report['rounds_completed']}；liveActor{report['live_Actor_Adam']}/V{report['live_V_Adam']}，相应shadow及独立历史重构PASS。完整开发结果见research_rescue_20261109/T441_FINAL_REPORT.json；fresh16/reload78未准入，six_gate=False、8未启动，不冒称稳定/晋升。remote={report['remote']}，全部科学进程退，safeT255冻结、Phase隔离。不重启或续旧预算。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report Stage T441 joint recurrent Actor Critic integration'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(210,end-time.time()-5))
    except BaseException:
        save(DOC/'T441_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),no_training_restart=True,raw_preserved=True,partial_report=report))
        # Deliberately do not re-raise into a sealed console file.

if __name__=='__main__':main()
