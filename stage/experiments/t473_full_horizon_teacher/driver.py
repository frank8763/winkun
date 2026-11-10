"""Finalize the frozen occupancy diagnostic once; no training launcher."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t473_full_horizon_teacher';DOC=ROOT/'stage/docs/overnight_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def call(*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'science.py'),*args],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'subprocesses.jsonl').open('a') as registry:
        registry.write(json.dumps(dict(pid=c.pid,pgid=c.pid,args=list(args),epoch=time.time()))+'\n')
        registry.flush();os.fsync(registry.fileno())
    try:
        b,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(b.decode(errors='replace')[-8000:])
    except BaseException:
        try:os.killpg(c.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:c.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(c.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            c.wait()
        raise

def scientific_stopped(folder):
    s=read(folder/'summary.json');generations=read(folder/'worker_generations.json') if (folder/'worker_generations.json').exists() else []
    ids=[s['pid']]+[pid for g in generations for pid in g['workers']]
    if (folder/'audit.json').exists():ids.append(read(folder/'audit.json')['pid'])
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    return ids

def all_stopped():
    registry=OUT/'subprocesses.jsonl';entries=[json.loads(x) for x in registry.read_text().splitlines()] if registry.exists() else []
    ids=[x['pid'] for x in entries];groups={x['pgid'] for x in entries}
    for folder in [OUT,OUT/'gate']:
        if (folder/'worker_generations.json').exists():
            for g in read(folder/'worker_generations.json'):ids+=[g['parent']]+g['workers']
        for name in ['summary.json','audit.json']:
            if (folder/name).exists():ids.append(read(folder/name)['pid'])
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail=stat.read_text().rsplit(')',1)[1].split()
            if int(tail[2]) in groups:ids.append(int(stat.parent.name))
        except (FileNotFoundError,ProcessLookupError):pass
    ids=sorted(set(ids));assert all(not Path('/proc',str(v)).exists() for v in ids),'Scientific process remains, no archive'
    return ids

def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];remote='PENDING';audit=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T473hardstop')))
    try:
        try:
            call('main',seconds=min(3003,end-time.time()-1980));s=read(OUT/'summary.json');scientific_stopped(OUT);assert s['state']=='COMPLETE'
            call('audit',seconds=min(900,end-time.time()-1080));checked_audit=read(OUT/'audit.json');assert checked_audit['state']=='PASS';audit=checked_audit
        except BaseException:save(DOC/'T473_SCIENTIFIC_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        ids=all_stopped();s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
        report=dict(state='COMPLETE_AUDITED_LOCAL' if audit else 'INCOMPLETE_PRESERVED',summary=s,audit=audit,
            diagnosis=read(OUT/'teacher_diagnosis.json') if (OUT/'teacher_diagnosis.json').exists() else None,
            remote='PENDING',no_training=True,six_gate=False,eight_started=False)
        save(OUT/'report_before_backup.json',report);save(DOC/'T473_FINAL_REPORT.json',report)
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time()))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),finalizer_writes_only_docs_after_this_point=True))
        c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T473'],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
        if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-5000:])
        remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T473.json')['state'];report['remote']=remote;save(DOC/'T473_FINAL_REPORT.json',report)
        save(DOC/'T473_STATUS.json',dict(state='FINISHED',epoch=time.time(),remote=remote))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10 T473旧15完整余时域原生相位CEM教师已结束，audit={bool(audit)}/remote={remote}，physics={s['physics'] if s else None}/commands={s['commands'] if s else None}；0Actor/optimizer，实际训练教师是否safe见报告，不能充自主六米成功；原T470/T472均未启动，无8，Phase隔离。详overnight_20261110/T473_FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/RESEARCH_STATUS.md','stage/docs/overnight_20261110','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report frozen counterexample recovery diagnosis'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
        if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
    except BaseException:save(DOC/'T473_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,remote=remote,no_restart=True))

if __name__=='__main__':main()
