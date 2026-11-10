"""Finite two-failure correction finalizer; always waits for user before8m."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t464_two_failure_recovery';DOC=ROOT/'stage/docs/review_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def call(*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),*args],cwd=ROOT,
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
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];gate=None;audit=None;summary=None;remote='PENDING';reason='TWO_FAILURE_FINITE_REPAIR_FINISHED'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T46420:00hardstop')))
    try:
        try:
            call('main',seconds=min(1803,p['absolute_native_stop']-time.time(),end-time.time()-2700))
            summary=read(OUT/'summary.json');scientific_stopped(OUT);assert summary['state']=='COMPLETE'
            call('audit',seconds=min(600,end-time.time()-2400));audit=read(OUT/'audit.json');assert audit['state']=='PASS'
        except BaseException:
            save(DOC/'T464_MAIN_OR_AUDIT_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
            if (OUT/'summary.json').exists():summary=read(OUT/'summary.json')
            reason='EVIDENCE_OR_RECOVERY_CONTRACT_STOP'
        if audit and summary and summary.get('candidate') and summary['candidate']['development_admitted']:
            now=time.time();folder=OUT/'gate';assert not folder.exists()
            assert now+1800+600+900<end and now+1800<p['absolute_native_stop']
            folder.mkdir();gp=dict(p)
            for key in ['inputs','constructor_reads']:gp.pop(key,None)
            gp.update(id='T464_INDEPENDENT_SIX_GATES',out=str(folder),learning_initial=summary['proposal'],
                max_bytes=p['gate_partition_bytes'],max_physics=4000000,max_commands=144,main_seconds=1800,learning_seconds=1,
                preflight_commands=0,model_bytes=2*2**20,baseline329=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines() if json.loads(x)['role']=='TWO_FAILURE_FINAL329'])
            save(folder/'protocol.json',gp);save(folder/'launch.json',dict(epoch=now,model=summary['proposal']))
            try:
                call('gatemain',seconds=1803);gate=read(folder/'summary.json');scientific_stopped(folder);assert gate['state']=='COMPLETE'
                call('gateaudit',seconds=min(600,end-time.time()-900));assert read(folder/'audit.json')['state']=='PASS'
                reason='SIX_NATIVE_GATES_PASS_PENDING_BINARY_BACKUP' if gate['native_gate'] else 'UNSEEN_OR_RELOAD_GATE_STOP_NO_ADAPTATION'
            except BaseException:
                reason='GATE_EVIDENCE_FAILURE_STOP';save(DOC/'T464_GATE_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
                if (folder/'summary.json').exists():gate=read(folder/'summary.json')
        elif audit and summary:
            reason='RECOVERY_GATE_FAILED_ZERO_BC' if not summary['gate']['all2_safe'] else 'DEVELOPMENT329_REMAINING_FAILURES'
        ids=all_stopped()
        gate_started=(OUT/'gate/launch.json').exists()
        ledger_complete=summary is not None and (not gate_started or gate is not None)
        known_rows=[]
        for folder in [OUT,OUT/'gate']:
            for name in ['cases.jsonl','replays.jsonl']:
                path=folder/name
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:known_rows.append(json.loads(line))
                        except json.JSONDecodeError:pass
        known_complete_physics_lower_bound=sum(row['physics'] for row in known_rows)
        physics=(summary['physics']+(gate['physics'] if gate else 0)) if ledger_complete else None
        commands=(summary['commands']+(gate['commands'] if gate else 0)) if ledger_complete else None
        if ledger_complete:
            assert physics<=p['overall_physics_max'] and commands<=p['overall_commands_max']
            assert p['prior_window_physics']+physics<=p['window_total_physics_max'] and p['prior_window_commands']+commands<=p['window_total_commands_max']
        else:reason='INCOMPLETE_LEDGER_UNKNOWN_COUNTS_NO_FURTHER_SCIENCE'
        report=dict(state='COMPLETE_AUDITED_LOCAL' if audit else 'STOPPED_RAW_PRESERVED',reason=reason,summary=summary,audit=audit,gate=gate,
            physics=physics,commands=commands,window_physics=(p['prior_window_physics']+physics if physics is not None else None),window_commands=(p['prior_window_commands']+commands if commands is not None else None),
            ledger_complete=ledger_complete,known_complete_physics_lower_bound=known_complete_physics_lower_bound,
            all_scientific_processes_stopped=True,remote='PENDING',six_gate=False,eight_started=False,user_review_required_before8=True)
        save(OUT/'report_before_backup.json',report);save(DOC/'T464_FINAL_REPORT.json',report)
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),finalizer_writes_only_docs_after_this_point=True,driver_pid_not_a_scientific_writer=os.getpid()))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock','.partial']]
        save(DOC/'T464_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T464'],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-6000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T464.json')['state']
        except BaseException:save(DOC/'T464_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,remote='PENDING'))
        gate_audit=read(OUT/'gate/audit.json') if (OUT/'gate/audit.json').exists() else None
        report['remote']=remote;report['six_gate']=bool(ledger_complete and audit and gate and gate.get('native_gate') and gate_audit and gate_audit['state']=='PASS' and remote=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
        report['state']='WAIT_USER_REVIEW_BEFORE_8M' if report['six_gate'] else report['state']
        save(DOC/'T464_FINAL_REPORT.json',report);save(DOC/'T464_STATUS.json',dict(state='FINISHED',reason=reason,epoch=time.time(),six_gate=report['six_gate'],eight_started=False,await_user_review=True))
        candidate=summary.get('candidate') if summary else None
        result='complete329notexecuted' if candidate is None else f"wide{candidate['wide54']['success']}/54old{candidate['old275']['success']}/275"
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10 T464两失败当前偏离修复已结束：{result}，reason={reason}，six_gate={report['six_gate']}，remote={remote}，physics={physics}/commands={commands}，累计原窗口{report['window_physics']}/{report['window_commands']}不扩30M/1600。4当前成功full/exact与2T461神经教师实际4sec恢复/exact，门过才128BC，不称新RL。结果等待用户看后再考虑8，8未启动/Phase隔离。详review_20261110/T464_FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/review_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report current two-failure Stage recovery and user review gate'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'T464_SOURCE_SYNC_ERROR.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'T464_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_science_restart=True,remote=remote,epoch=time.time()))

if __name__=='__main__':main()
