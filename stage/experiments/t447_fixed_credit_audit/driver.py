"""Finite no-update audit and stopped-byte backup; no navigation process."""
from pathlib import Path
import json,os,subprocess,time,traceback,signal
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t447_fixed_credit_audit';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(f):return json.loads(f.read_text())
def save(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        b,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(b.decode(errors='replace')[-6000:])
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
    try:
        child('run.py','main',seconds=min(900,end-time.time()-1200));child('run.py','audit',seconds=min(900,end-time.time()-480))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS_FROZEN_DIAGNOSIS_ONLY';ids=[s['pid'],a['pid']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',summary=s,diagnosis=read(OUT/'diagnosis.json'),remote='PENDING',new_navigation_training=0,six_gate=False,eight_started=False)
        save(DOC/'T447_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        try:child('transport.py','T447',seconds=min(300,end-time.time()-180));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T447.json')['state']
        except BaseException:save(DOC/'T447_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True))
        save(DOC/'T447_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text(f"2026-11-09 T447冻结219成功history误差/10任务梯度审计完成，0native0Adam/独立10梯度及分段误差重构PASS；remote={report['remote']}，仅局部一阶诊断非有限更新/闭环收益，详research_rescue_20261109/T447_FINAL_REPORT。没有新策略/未过six门/8未进，safeT255及原失败不改，Phase隔离，不自动续旧训练。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','commit','-m','Report fixed-history Stage task-gradient and action-error diagnosis'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(160,end-time.time()-5))
    except BaseException:save(DOC/'T447_DRIVER_ERROR.json',dict(error=traceback.format_exc(),native=0,Adam=0,no_restart=True,epoch=time.time()))
if __name__=='__main__':main()
