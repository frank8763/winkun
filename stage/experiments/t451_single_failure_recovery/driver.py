"""Unique bounded finalizer; no restart or hidden training continuation."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t451_single_failure_recovery';DOC=ROOT/'stage/docs/review_20261110'
def read(f):return json.loads(f.read_text())
def save(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def child(script,*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        b,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(b.decode(errors='replace')[-6500:])
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
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end));signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T451 hardstop')))
    try:
        try:child('run.py','main',seconds=min(p['main_seconds']+5,end-time.time()-2200))
        except BaseException:save(DOC/'T451_MAIN_ERROR.json',dict(error=traceback.format_exc(),no_restart=True,raw_preserved=True))
        s=read(OUT/'summary.json');ids=[s['pid']]+[v for g in read(OUT/'worker_generations.json') for v in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids);audit=None
        if s['state']=='COMPLETE':
            try:child('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-900));audit=read(OUT/'audit.json');ids.append(audit['pid'])
            except BaseException:save(DOC/'T451_AUDIT_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_repeat=True))
        assert all(not Path('/proc',str(pid)).exists() for pid in ids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.lock','.partial','.pyc']];save(DOC/'T451_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        report=dict(state='COMPLETE_AUDITED_LOCAL' if audit else 'STOPPED_NOT_FULL_AUDIT_PASS',summary=s,audit=audit,remote='PENDING',six_gate=False,eight_started=False);save(DOC/'T451_FINAL_REPORT.json',report)
        try:child('transport.py','T451',seconds=min(620,end-time.time()-200));report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T451.json')['state']
        except BaseException:save(DOC/'T451_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report['six_gate']=bool(audit and s.get('gate') and s['gate'].get('native_gate') and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED');save(DOC/'T451_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10 T451单例实际恢复修复已收尾，state={report['state']}，remote={report['remote']}，physics={s['physics']} commands={s['commands']}；详review_20261110/T451_FINAL_REPORT。six_gate={report['six_gate']}，8未启动，safeT255冻结/Phase隔离，不重启或追加旧预算。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/review_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','commit','-m','Report bounded single-failure Stage recovery and complete navigation gates'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:child('transport.py','source_sync',seconds=min(180,end-time.time()-5))
        except BaseException:save(DOC/'T451_SOURCE_SYNC_ERROR.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'T451_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()))
if __name__=='__main__':main()
