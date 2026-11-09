"""One finite no-learning diagnostic and its sole finalizer."""
from pathlib import Path
import json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t444_paired_retention';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        out,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(script+' exit '+str(c.returncode)+': '+out.decode(errors='replace')[-6000:])
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
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop']
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T444 supervisor stop')))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    try:
        child('run.py','main',seconds=min(p['main_seconds']+5,end-time.time()-1100))
        child('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-810))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
        ids=[s['pid'],a['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',summary=s,comparisons=read(OUT/'comparisons.json'),
            remote='PENDING',six_gate=False,eight_started=False,Actor_Adam=0,V_Adam=0)
        save(DOC/'T444_FINAL_REPORT.json',report)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        try:
            child('transport.py','T444',seconds=min(620,end-time.time()-200))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T444.json')['state']
        except BaseException:save(DOC/'T444_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T444_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text(f"2026-11-09 T444冻结配对诊断已结束/独立auditPASS，{s['primary']}不同policy-job原始轨迹+{s['exact']}exact，{s['physics']}物理/{s['commands']}全部命令；0Actor/V/Adam，safe成功{s['safe_success']}/8，当前候选碰撞{s['student_collisions']}/8。并非新学习/完整验收，six_gate=False/8未进；两策略各自h0重建同history，分解仅代数非唯一因果，详research_rescue_20261109/T444_FINAL_REPORT。remote={report['remote']}，原模型/失败保持/Phase隔离，不重启。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report frozen Stage T444 paired policy and history diagnosis'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(180,end-time.time()-5))
    except BaseException:save(DOC/'T444_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_training_restart=True,epoch=time.time()))
if __name__=='__main__':main()
