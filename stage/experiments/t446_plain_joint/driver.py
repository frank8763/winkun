"""Single bounded offline scientific driver and sole stopped-run finalizer."""
from pathlib import Path
import json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t446_plain_joint';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(f):return json.loads(f.read_text())
def save(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    p=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        b,_=p.communicate(timeout=max(1,seconds))
        if p.returncode:raise RuntimeError(script+' exit'+str(p.returncode)+': '+b.decode(errors='replace')[-6000:])
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
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T446 hardstop')))
    try:
        child('run.py','main',seconds=min(5405,end-time.time()-2700));child('run.py','audit',seconds=min(1800,end-time.time()-900))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS';ids=[s['pid'],a['pid']]+read(OUT/'workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',summary=s,remote='PENDING',fresh16=None,reload78=None,six_gate=False,eight_started=False,safeT255_not_promoted_or_modified=True,standard_BC_GRU_PPO_NOT_claimed_original=True)
        save(DOC/'T446_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        try:
            child('transport.py','T446',seconds=min(620,end-time.time()-190));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T446.json')['state']
        except BaseException:save(DOC/'T446_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T446_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-09 T446干净单Actor联合迁移试验已结束/独立auditPASS，BC{s['BC_live_Adam']}live+同shadow，PPO完成{len(s['rounds'])}轮，physics{s['physics']}/commands{s['commands']}，完整结果见research_rescue_20261109/T446_FINAL_REPORT；remote={report['remote']}，fresh16/reload78未准入/six_gateFalse/8未起，模型研究不晋升，Phase隔离，不重启或扩大预算。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report bounded plain Stage Actor joint transfer integration'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child('transport.py','source_sync',seconds=min(180,end-time.time()-5))
    except BaseException:save(DOC/'T446_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()))
if __name__=='__main__':main()
