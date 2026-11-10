"""Bounded local relay. No network/LLM dependency for any scientific stage."""
from pathlib import Path
import copy,hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t477_wide8_recovery';DOC=ROOT/'stage/docs/overnight_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def call(action,mode,seconds):
    assert seconds>0;c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t477_runtime.py'),action,mode],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'subprocesses.jsonl').open('a') as f:f.write(json.dumps(dict(pid=c.pid,pgid=c.pid,action=action,mode=mode,epoch=time.time()))+'\n');f.flush();os.fsync(f.fileno())
    try:
        b,_=c.communicate(timeout=seconds)
        if c.returncode:raise RuntimeError(b.decode(errors='replace')[-10000:])
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
        if (folder/name).exists():ids.append(read(folder/name)['pid'])
    if (folder/'worker_generations.json').exists():
        for g in read(folder/'worker_generations.json'):ids+=[g['parent']]+g['workers']
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    return ids

def manifest(folder):
    stopped(folder);files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(folder/'local_manifest.json',dict(files=files,remote='PENDING'));assert all(sha(ROOT/v['path'])==v['sha256'] for v in files)

def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];nativeend=p['absolute_native_stop'];learnend=p['absolute_learning_stop'];completed=[];reason='FINITE_THREE_MECHANISMS_COMPLETE';remote='PENDING';physics=0;commands=read(OUT/'preflight.json')['commands'];main_s=0.;allpids=[]
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,local_no_network=True))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T477 hard09:00 stop')))
    try:
        for mode in ['baseline',*p['modes']]:
            branch=mode!='baseline';reservephysics=15000000 if branch else 54*53334;reservecmd=520 if branch else 58;mainlimit=9000 if branch else 1800;auditlimit=1200 if branch else 300
            now=time.time()
            usedbytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
            if branch and usedbytes+15*2**20>p['max_bytes']:
                reason='COMPLETE_BRANCH_MINIMUM_AND_FINAL383_CAPACITY_STOP';break
            if physics+reservephysics>p['root_max_physics'] or commands+reservecmd>p['root_max_commands'] or main_s+mainlimit>p['root_max_main_seconds'] or now+mainlimit>nativeend or now+mainlimit+auditlimit+900>end or (branch and now+6000>learnend):
                reason='FULL_WORST_REMAINING_ADMISSION_STOP';break
            folder=OUT/mode;assert not folder.exists();folder.mkdir();pp=copy.deepcopy(p);pp.update(out=str(folder),mode=mode,max_physics=reservephysics,max_commands=reservecmd,main_seconds=mainlimit)
            if mode=='baseline':pp.update(panel_jobs=p['wide8_jobs'],learning_seconds=1)
            if mode=='longcredit':pp['lambda_native']=1.
            save(folder/'protocol.json',pp);save(folder/'launch.json',dict(epoch=now,mode=mode,initial=p['initial'],independent_arm=True))
            save(OUT/'status.json',dict(state='RUNNING_'+mode,epoch=now,completed=completed,physics=physics,commands=commands,all_stop=end))
            try:
                call('main',mode,min(mainlimit+3,nativeend-time.time()-5));s=read(folder/'summary.json');allpids+=stopped(folder);assert s['state']=='COMPLETE'
                physics+=s['physics'];commands+=s['commands'];main_s+=s['elapsed_s'];assert physics<=p['root_max_physics'] and commands<=p['root_max_commands']
                call('audit',mode,min(auditlimit,end-time.time()-900));a=read(folder/'audit.json');assert a['state']=='PASS';allpids+=stopped(folder);manifest(folder)
                completed.append(mode);save(DOC/'T477_LEDGER.json',dict(completed=completed,physics=physics,commands=commands,main_seconds=main_s,all_stop=end,latest=s))
            except BaseException:
                save(DOC/'T477_SCIENTIFIC_ERROR.json',dict(mode=mode,error=traceback.format_exc(),all_raw_preserved=True,no_restart=True));reason='EVIDENCE_OR_ENGINEERING_FAILURE_STOP';break
        # Parent groups, all workers and audits must really be gone before seal.
        for f in OUT.glob('*/worker_generations.json'):allpids+=stopped(f.parent)
        groups={x['pgid'] for x in map(json.loads,(OUT/'subprocesses.jsonl').read_text().splitlines())} if (OUT/'subprocesses.jsonl').exists() else set()
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:
                z=stat.read_text().rsplit(')',1)[1].split()
                assert int(z[2]) not in groups,('Scientific descendant still alive',stat)
            except (FileNotFoundError,ProcessLookupError):pass
        summaries={f.parent.name:read(f) for f in OUT.glob('*/summary.json')}
        # Reconcile failures too; missing values stay null and never become rates.
        knownphysics=sum(s['physics'] for s in summaries.values());knowncommands=read(OUT/'preflight.json')['commands']+sum(s['commands'] for s in summaries.values());assert knownphysics<=p['root_max_physics'] and knowncommands<=p['root_max_commands']
        report=dict(state='COMPLETE_AUDITED_LOCAL' if len(completed)==4 else 'BOUNDED_QUEUE_STOPPED',reason=reason,completed=completed,summaries=summaries,physics=knownphysics,commands=knowncommands,main_seconds=sum(s['elapsed_s'] for s in summaries.values()),remote='PENDING',six_gate=False,eight_stability_gate=False,fresh16=None,reload78=None,inherited_old00_failure=True,all_scientific_processes_stopped=True)
        save(OUT/'report_before_backup.json',report);save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(allpids)),epoch=time.time(),finalizer_writes_only_docs_after_this_point=True))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock']];save(DOC/'T477_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T477'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-6000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T477.json')['state']
        except BaseException:save(DOC/'T477_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report['remote']=remote;save(DOC/'T477_FINAL_REPORT.json',report)
        lines=['# T477 wide8 actual results',f"State: {report['state']}; reason: {reason}; remote: {remote}",f"Physical: {knownphysics}; commands: {knowncommands}; completed: {completed}",'Initial T449 wide6 54/54, old274/275 inherited failure dev_task6_00. No full6 gate or new8stability claim. No fresh16/reload78 allocated; not executed.']
        for mode,s in summaries.items():
            c=s.get('candidate');lines.append(f"{mode}: state={s['state']}; physical={s['physics']}; commands={s['commands']}; candidate="+json.dumps({k:v for k,v in (c or {}).items() if k in ['wide8','wide6','old275','counts']},ensure_ascii=False))
        save(DOC/'T477_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,remote=remote,no_automatic_relaunch=True));(DOC/'T477_FINAL_REPORT.md').write_text('\n\n'.join(lines)+'\n')
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10/11 T477宽八米有限离线接力结束，completed={completed}/reason={reason}/physics={knownphysics}/commands={knowncommands}/remote={remote}；旧唯一失败继承不冒称修复，fresh16/reload78未执行，六米完整门与八米稳定门未声明通过。详overnight_20261110/T477_FINAL_REPORT，Phase完全隔离。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/overnight_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report bounded wide8 recurrent PPO relay and verified evidence'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'T477_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'T477_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_scientific_retry=True,raw_preserved=True))
if __name__=='__main__':main()
