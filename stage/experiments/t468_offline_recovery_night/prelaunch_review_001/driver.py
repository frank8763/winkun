"""Independent local finite night relay; no network until science has stopped."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t468_offline_recovery_night';DOC=ROOT/'stage/docs/overnight_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
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

def stopped(folder):
    s=read(folder/'summary.json');ids=[s['pid']]
    if (folder/'worker_generations.json').exists():
        for g in read(folder/'worker_generations.json'):ids+=[g['parent']]+g['workers']
    if (folder/'audit.json').exists():ids.append(read(folder/'audit.json')['pid'])
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    return ids

def seal_child(folder):
    ids=stopped(folder);save(folder/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.partial','.pyc']]
    save(folder/'local_manifest.json',dict(files=files,remote='PENDING'))
    assert all(sha(ROOT/v['path'])==v['sha256'] for v in files)

def all_stopped():
    entries=[json.loads(x) for x in (OUT/'subprocesses.jsonl').read_text().splitlines()] if (OUT/'subprocesses.jsonl').exists() else []
    groups={v['pgid'] for v in entries};ids=[v['pid'] for v in entries]
    for folder in [v for v in OUT.iterdir() if v.is_dir() and (v.name.startswith(('occupancy_','duration_')) or v.name=='gate')]:
        for name in ['summary.json','audit.json']:
            if (folder/name).exists():ids.append(read(folder/name)['pid'])
        if (folder/'worker_generations.json').exists():
            for g in read(folder/'worker_generations.json'):ids+=[g['parent']]+g['workers']
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=stat.read_text().rsplit(')',1)[1].split()
            if int(fields[2]) in groups:ids.append(int(stat.parent.name))
        except (FileNotFoundError,ProcessLookupError):pass
    ids=sorted(set(ids));assert all(not Path('/proc',str(v)).exists() for v in ids),'Unstopped science chain: no seal/archive'
    return ids

def main():
    import run as X
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];current=p['initial_candidate'];ledger=[];completed=[]
    physics=0;commands=p['preflight_commands'];elapsed=0.;gate=None;gate_audit=None;all_audited=True;remote='PENDING';reason='FINITE_FIXED_NIGHT_MECHANISMS_COMPLETE'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T46808:30allstop')))
    try:
        stopchain=False
        for mode in p['modes']:
            for index in range(p['max_rounds_per_mode']):
                now=time.time();folder=OUT/f'{mode}_{index}'
                admitted=(now+2400+600+1800+600+900<end and now+2400+1800<p['absolute_native_stop'] and now+1200<p['absolute_learning_stop']
                    and physics+8000000+4000000<=p['max_physics'] and commands+400+160<=p['max_commands']
                    and elapsed+2400+1800<=p['max_main_seconds']
                    and used(OUT)+p['round_partition_bytes']+p['gate_partition_bytes']+2**20<=p['max_bytes'])
                if not admitted:reason='NEXT_ROUND_WHOLE_PHASE_PLUS_GATE_BUDGET_NOT_ADMITTED';stopchain=True;break
                assert not folder.exists();folder.mkdir();cp=X.child_config(p,mode,index,current)
                cp['main_seconds']=min(2400,int(p['absolute_native_stop']-now));cp['learning_seconds']=min(1200,int(p['absolute_learning_stop']-now))
                save(folder/'protocol.json',cp);save(folder/'launch.json',dict(epoch=now,mode=mode,index=index,input_model=current['model']))
                save(OUT/'status.json',dict(state=f'{mode.upper()}_{index}_RUNNING',epoch=time.time(),driver=os.getpid()))
                try:
                    call('roundmain',mode,str(index),seconds=cp['main_seconds']+3)
                    s=read(folder/'summary.json');stopped(folder);assert s['state']=='COMPLETE'
                    call('roundaudit',mode,str(index),seconds=min(600,end-time.time()-900));assert read(folder/'audit.json')['state']=='PASS'
                    seal_child(folder)
                except BaseException:
                    save(folder/'driver_failure.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));all_audited=False
                    if (folder/'summary.json').exists():
                        s=read(folder/'summary.json');physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
                    else:physics=commands=None
                    completed.append(folder.name);reason='SHARED_EVIDENCE_OR_NUMERIC_FAILURE_STOP';stopchain=True;break
                completed.append(folder.name);physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
                candidate=s.get('candidate');accept=bool(candidate and set(current['success_ids'])<=set(candidate['success_ids']) and len(candidate['success_ids'])>len(current['success_ids']))
                ledger.append(dict(mode=mode,index=index,candidate=candidate,accepted=accept,input_model=current['model'],
                    physics=s['physics'],commands=s['commands'],elapsed_s=s['elapsed_s'],coverage_gate=s.get('coverage'),
                    reuse=s.get('reused_recovery_rows'),completed_paired_BC=s.get('completed_paired_BC')))
                save(OUT/'ledger.json',ledger)
                assert physics<=p['max_physics'] and commands<=p['max_commands'] and elapsed<=p['max_main_seconds']
                if accept:current=candidate
                if candidate and candidate['development_admitted']:
                    assert accept;gfolder=OUT/'gate';assert not gfolder.exists();gfolder.mkdir();gp=X.child_config(p,mode,index,current)
                    now=time.time();assert now+1800+600+900<end and now+1800<p['absolute_native_stop']
                    gp.update(id='T468_INDEPENDENT_SIX_GATES',out=str(gfolder),max_bytes=p['gate_partition_bytes'],max_physics=4000000,max_commands=160,
                        main_seconds=1800,learning_seconds=1,model_bytes=2*2**20,baseline329=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines() if json.loads(x)['role']=='FINAL329'])
                    save(gfolder/'protocol.json',gp);save(gfolder/'launch.json',dict(epoch=now,model=current['model']))
                    try:
                        call('gatemain',seconds=1803);gate=read(gfolder/'summary.json');stopped(gfolder);assert gate['state']=='COMPLETE'
                        call('gateaudit',seconds=min(600,end-time.time()-900));gate_audit=read(gfolder/'audit.json');assert gate_audit['state']=='PASS';seal_child(gfolder)
                    except BaseException:
                        all_audited=False;save(gfolder/'driver_failure.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
                        if (gfolder/'summary.json').exists():gate=read(gfolder/'summary.json')
                    if gate:physics+=gate['physics'];commands+=gate['commands'];elapsed+=gate['elapsed_s']
                    else:physics=commands=None
                    completed.append('gate');reason='SIX_NATIVE_GATES_PASS_PENDING_BACKUP_USER_REVIEW' if gate and gate.get('native_gate') else 'FRESH_OR_RELOAD_GATE_STOP_NO_ADAPTATION'
                    stopchain=True;break
                if not accept:
                    # Repeating same-policy failed data cannot be called a new
                    # occupancy round. Move only to the next distinct mechanism.
                    break
            if stopchain:break
        ids=all_stopped()
        if physics is not None:assert physics<=p['max_physics'] and commands<=p['max_commands'] and elapsed<=p['max_main_seconds']
        else:all_audited=False
    except BaseException:
        reason='DRIVER_OR_UNFINISHED_LEDGER_STOP_PRESERVE_ALL_RAW';all_audited=False
        save(DOC/'DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()))
        ids=all_stopped()
    try:
        report=dict(state='COMPLETE_AUDITED_LOCAL' if all_audited else 'STOPPED_RAW_PRESERVED',reason=reason,completed=completed,ledger=ledger,
            current=current,gate=gate,gate_audit=gate_audit,physics=physics,commands=commands,main_elapsed_s=elapsed,
            six_gate=False,eight_started=False,user_review_required_before8=True,remote='PENDING',all_scientific_processes_stopped=True,
            missing_counts_are_unknown_notzero=physics is None)
        save(OUT/'summary.json',dict(state='SEALED_FINITE_NIGHT_ROOT',pid=os.getpid(),physics=physics,commands=commands,main_elapsed_s=elapsed,reason=reason,completed=completed))
        save(OUT/'report_before_backup.json',report);save(DOC/'FINAL_REPORT.json',report)
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),finalizer_writes_only_docs_after_this_point=True,driver_pid_not_a_scientific_writer=os.getpid()))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock','.partial']]
        save(DOC/'LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T468'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-6000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T468.json')['state']
        except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True,remote='PENDING'))
        report['remote']=remote;report['six_gate']=bool(all_audited and gate and gate.get('native_gate') and gate_audit and gate_audit['state']=='PASS' and remote=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
        if report['six_gate']:report['state']='WAIT_USER_REVIEW_BEFORE_8M'
        save(DOC/'FINAL_REPORT.json',report);save(DOC/'NIGHT_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,six_gate=report['six_gate'],remote=remote,eight_started=False))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10夜间T468有限离线纠正接力已结束，completed={completed}/reason={reason}，physics={physics}/commands={commands}，remote={remote}/six_gate={report['six_gate']}。实际占用与完整时长目标两机制各至多2轮，拒绝不传不重复同策略占用，8等待用户看后，本程序未启动8。所有raw/模型/失败保留，Phase隔离。详overnight_20261110/FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/overnight_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report finite offline Stage night recovery and actual gates'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'SOURCE_REMOTE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'FINALIZATION_ERROR.json',dict(error=traceback.format_exc(),remote=remote,raw_preserved=True,no_science_restart=True))

if __name__=='__main__':main()
