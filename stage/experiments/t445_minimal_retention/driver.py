"""One no-training minimal diagnosis and finalizer within original T444 stop."""
from pathlib import Path
import sys,json,os,time,subprocess,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
import signal
OUT=ROOT/'stage/runs/t445_minimal_retention';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        out,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(out.decode(errors='replace')[-6000:])
    except BaseException:
        try:os.killpg(c.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:c.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(c.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            c.wait()
        raise
def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T445 hardstop')))
    try:
        child('run.py','main',seconds=min(540,end-time.time()-900));child('run.py','audit',seconds=min(180,end-time.time()-640))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS';ids=[s['pid'],a['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',summary=s,comparisons=read(OUT/'comparisons.json'),remote='PENDING',
            combined_original_failure_physics=read(ROOT/'stage/runs/t444_paired_retention/summary.json')['physics']+s['physics'],
            six_gate=False,eight_started=False,Actor_Adam=0,V_Adam=0,scope='Two representative oldLEFT5 cases, not complete eight-case or navigation acceptance')
        save(DOC/'T445_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        try:
            child('transport.py','T445',seconds=min(600,end-time.time()-150));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T445.json')['state']
        except BaseException:save(DOC/'T445_BACKUP_ERROR.json',dict(error=traceback.format_exc(),no_restart=True))
        save(DOC/'T445_FINAL_REPORT.json',report)
        st=ROOT/'stage/docs/RESEARCH_STATUS.md';st.write_text(f"2026-11-09 T445最小两配对冻结诊断已结束/独立auditPASS，原T444工程失败不改，合计{report['combined_original_failure_physics']}物理；本轮{s['physics']}物理/{s['commands']}命令，4原始+4exact、0训练。两例结果见research_rescue_20261109/T445_FINAL_REPORT，非8例全部定位/非导航验收，six_gateFalse/8未进；remote={report['remote']}，Phase隔离/模型不改。\n\n"+st.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report minimal frozen Stage policy retention diagnosis'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(120,end-time.time()-5))
    except BaseException:save(DOC/'T445_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()))
if __name__=='__main__':main()
