"""One local offline finalizer; fixed finite queue, strict independent gates."""
from pathlib import Path
import copy,hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t475_direct_observation_night';DOC=ROOT/'stage/docs/overnight_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def call(*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t475_runtime.py'),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'subprocesses.jsonl').open('a') as f:f.write(json.dumps(dict(pid=c.pid,pgid=c.pid,args=args,epoch=time.time()))+'\n');f.flush();os.fsync(f.fileno())
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

def stopped(folder):
    ids=[]
    for name in ['summary.json','audit.json']:
        f=folder/name
        if f.exists():ids.append(read(f)['pid'])
    f=folder/'worker_generations.json'
    if f.exists():
        for g in read(f):ids+=[g['parent']]+g['workers']
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    return ids

def seal(folder):
    ids=stopped(folder);save(folder/'SEALED.json',dict(scientific_pids=ids,all_scientific_processes_stopped=True,epoch=time.time()))
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.pyc','.lock']]
    save(folder/'local_manifest.json',dict(files=files,remote='PENDING'));assert all(sha(ROOT/v['path'])==v['sha256'] for v in files)

def all_stopped():
    rows=[json.loads(x) for x in (OUT/'subprocesses.jsonl').read_text().splitlines()] if (OUT/'subprocesses.jsonl').exists() else []
    groups={v['pgid'] for v in rows};ids=[v['pid'] for v in rows]
    for folder in [OUT,OUT/'gate']:ids+=stopped(folder)
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=stat.read_text().rsplit(')',1)[1].split()
            if int(fields[2]) in groups:ids.append(int(stat.parent.name))
        except (FileNotFoundError,ProcessLookupError):pass
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    return sorted(set(ids))

def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];s=None;audit=None;gate=None;gate_audit=None;remote='PENDING';reason='FINAL_ATTEMPT_FAILED_STOP_STAGE';physics=commands=None
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T475 final attempt hardstop')))
    try:
        try:
            call('main',seconds=min(4203,end-time.time()-3300));s=read(OUT/'summary.json');stopped(OUT);assert s['state']=='COMPLETE'
            call('audit',seconds=min(1200,end-time.time()-2100));checked=read(OUT/'audit.json');assert checked['state']=='PASS';audit=checked
        except BaseException:
            reason='FINAL_ATTEMPT_EVIDENCE_FAILURE_STOP_STAGE';save(DOC/'T475_SCIENTIFIC_ERROR.json',dict(error=traceback.format_exc(),no_retry=True,raw_preserved=True))
            if (OUT/'summary.json').exists():s=read(OUT/'summary.json')
        if audit and s['candidate']['development_admitted']:
            folder=OUT/'gate';folder.mkdir();gp=copy.deepcopy(p);gp.update(id='T475_INDEPENDENT_SIX_GATES',out=str(folder),gate_model=s['proposal'],max_bytes=p['gate_partition_bytes'],max_physics=3600000,max_commands=102,main_seconds=2400,learning_seconds=1,
                baseline329=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()])
            save(folder/'protocol.json',gp);save(folder/'launch.json',dict(epoch=time.time(),model=s['proposal']))
            try:
                call('gatemain',seconds=min(2403,end-time.time()-1500));gate=read(folder/'summary.json');stopped(folder);assert gate['state']=='COMPLETE'
                call('gateaudit',seconds=min(600,end-time.time()-900));checked=read(folder/'audit.json');assert checked['state']=='PASS';gate_audit=checked
                reason='FINAL_ATTEMPT_NATIVE_GATES_PASS_PENDING_BACKUP_USER_REVIEW' if gate['native_gate'] else 'FINAL_ATTEMPT_FRESH_OR_RELOAD_FAILED_STOP_STAGE'
            except BaseException:
                reason='FINAL_ATTEMPT_GATE_EVIDENCE_FAILURE_STOP_STAGE';save(DOC/'T475_GATE_ERROR.json',dict(error=traceback.format_exc(),no_retry=True,raw_preserved=True))
                if (folder/'summary.json').exists():gate=read(folder/'summary.json')
        ids=all_stopped();complete=s is not None and (not (OUT/'gate').exists() or gate is not None)
        if complete:
            physics=s['physics']+(gate['physics'] if gate else 0);commands=p['preflight_commands']+s['commands']+(gate['commands'] if gate else 0)
            assert physics<=p['overall_physics_max'] and commands<=441 and p['night_prior_commands']+commands<=2000 and p['night_prior_physics']+physics<=40000000
        report=dict(state='COMPLETE_AUDITED_LOCAL' if audit else 'STOPPED_RAW_PRESERVED',reason=reason,summary=s,audit=audit,gate=gate,gate_audit=gate_audit,physics=physics,commands=commands,
            night_physics=p['night_prior_physics']+physics if physics is not None else None,night_commands=p['night_prior_commands']+commands if commands is not None else None,
            remote='PENDING',six_gate=False,final_attempt=True,no_further_stage_experiments=True,eight_started=False,all_scientific_processes_stopped=True)
        save(OUT/'report_before_backup.json',report);save(DOC/'T475_FINAL_REPORT.json',report);save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),finalizer_writes_only_docs_after_this_point=True))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock']];save(DOC/'T475_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T475'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-6000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T475.json')['state']
        except BaseException:save(DOC/'T475_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report['remote']=remote;report['six_gate']=bool(audit and gate and gate.get('native_gate') and gate_audit and gate_audit['state']=='PASS' and remote=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
        report['state']='WAIT_USER_REVIEW_BEFORE_8M' if report['six_gate'] else report['state'];report['stage_line_stopped']=not report['six_gate']
        save(DOC/'T475_FINAL_REPORT.json',report);save(DOC/'T475_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,six_gate=report['six_gate'],remote=remote,stage_line_stopped=not report['six_gate'],no_automatic_new_experiments=True,eight_started=False))
        c=s.get('candidate') if s else None;result=f"wide{c['wide54']['success']}/54 old{c['old275']['success']}/275" if c else 'NOT_EXECUTED'
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10 T475最后一次Stage试验结束，{result}/reason={reason}/physics={physics}/commands={commands}/remote={remote}/six_gate={report['six_gate']}。用户要求未通过即停止，本管线无后续新实验/不再自动重试；原327/safeT255/全部证据保留，无8，Phase隔离。详overnight_20261110/T475_FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/overnight_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report final bounded Stage attempt and honor stop decision'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'T475_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'T475_FINALIZER_ERROR.json',dict(error=traceback.format_exc(),no_scientific_retry=True,raw_preserved=True))

if __name__=='__main__':main()
