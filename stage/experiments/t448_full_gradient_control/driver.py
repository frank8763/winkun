"""One finite paired control; no planner or autonomous continuation grid."""
from pathlib import Path
import json,os,subprocess,signal,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t448_full_gradient_control';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    p=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        b,_=p.communicate(timeout=max(1,seconds))
        if p.returncode:raise RuntimeError(b.decode(errors='replace')[-6000:])
    except BaseException:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
        raise
def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T448 hardstop')))
    try:
        child('run.py','main',seconds=min(3605,end-time.time()-2200));child('run.py','audit',seconds=min(1200,end-time.time()-900))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS';ids=[s['pid'],a['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',summary=s,remote='PENDING',PPO_updates=0,models_research_NOT_promoted=True,six_gate=False,eight_started=False)
        save(DOC/'T448_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        try:child('transport.py','T448',seconds=min(620,end-time.time()-180));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T448.json')['state']
        except BaseException:save(DOC/'T448_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T448_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-09 T448固定全数据梯度vs循环小批量对照已结束/独立128历史Adam重构PASS，live{s['live_Actor_Adam']}+shadow{s['shadow_Actor_Adam']}，0PPO/V，{s['physics']}物理/{s['commands']}全部命令；两末模型各完整329，结果见research_rescue_20261109/T448_FINAL_REPORT。remote={report['remote']}；每步数据/计算量不同，不作效率/纯噪声因果或创新声称，fresh16/reload78未执行/six_gateFalse/8未进，safeT255不改/Phase隔离，不重启或追加预算。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','commit','-m','Report paired finite full-empirical-gradient Stage transfer control'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(160,end-time.time()-5))
    except BaseException:save(DOC/'T448_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()))
if __name__=='__main__':main()
