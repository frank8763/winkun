"""One local offline finalizer; fixed finite queue, strict independent gates."""
from pathlib import Path
import copy,hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t474_verified_joint_night';DOC=ROOT/'stage/docs/overnight_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def call(*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t474_runtime.py'),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
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
    for folder in [OUT/n for n in ['verified','joint','priority','gate']]:ids+=stopped(folder)
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=stat.read_text().rsplit(')',1)[1].split()
            if int(fields[2]) in groups:ids.append(int(stat.parent.name))
        except (FileNotFoundError,ProcessLookupError):pass
    assert all(not Path('/proc',str(v)).exists() for v in ids)
    return sorted(set(ids))

def main():
    import control as C
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];physics=0;commands=p['preflight_commands'];elapsed=0.;current=p['initial_candidate'];completed=[];ledger=[];audited=True;verification=None;gate=None;gate_audit=None;remote='PENDING';reason='FIXED_FINITE_MECHANISMS_COMPLETE'
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T47408:30hardstop')))
    try:
        folder=OUT/'verified';folder.mkdir();vp=C.verify_config(p);save(folder/'protocol.json',vp)
        save(OUT/'status.json',dict(state='FROZEN_CONTROL_INDEPENDENT_REPLAY',epoch=time.time()))
        call('verify_main',seconds=603);s=read(folder/'summary.json');stopped(folder);assert s['state']=='COMPLETE'
        physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
        call('verify_audit',seconds=300);checked=read(folder/'audit.json');assert checked['state']=='PASS';verification=checked;seal(folder)
        completed.append('verified');rawrow=checked['verified_raw15']
        for mode in p['modes']:
            now=time.time()
            if not(now+2400+600+1800+600+900<end and now+2400+600+1800<p['absolute_native_stop'] and now+1200<p['absolute_learning_stop']
                and physics+8000000+4000000<=p['max_physics'] and commands+345+126<=p['max_commands'] and elapsed+2400+1800<=p['max_main_seconds']
                and used(OUT)+p['train_partition_bytes']+p['gate_partition_bytes']+2**20<=p['max_bytes']):
                reason='WHOLE_NEXT_BRANCH_PLUS_GATE_NOT_ADMITTED';break
            folder=OUT/mode;folder.mkdir();cp=C.train_config(p,mode,rawrow);save(folder/'protocol.json',cp);save(folder/'launch.json',dict(epoch=now,input_model=cp['learning_initial']))
            save(OUT/'status.json',dict(state=mode.upper()+'_LEARNING_AND_FULL329',epoch=time.time()))
            call('train_main',mode,seconds=2403);s=read(folder/'summary.json');stopped(folder);assert s['state']=='COMPLETE'
            physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
            call('train_audit',mode,seconds=600);checked=read(folder/'audit.json');assert checked['state']=='PASS';seal(folder);completed.append(mode)
            candidate=s['candidate'];accepted=bool(set(current['success_ids'])<=set(candidate['success_ids']) and len(candidate['success_ids'])>len(current['success_ids']))
            ledger.append(dict(mode=mode,accepted=accepted,candidate=candidate,physics=s['physics'],commands=s['commands'],elapsed_s=s['elapsed_s'],live_Adam=128,shadow_Adam=128,input_always_original327=True));save(OUT/'ledger.json',ledger)
            if accepted:current=candidate
            assert physics<=p['max_physics'] and commands<=p['max_commands'] and elapsed<=p['max_main_seconds']
            if candidate['development_admitted']:
                assert accepted;folder=OUT/'gate';folder.mkdir();gp=copy.deepcopy(cp);gp.update(id='T474_INDEPENDENT_SIX_GATES',out=str(folder),learning_initial=candidate['model'],max_bytes=p['gate_partition_bytes'],max_physics=4000000,max_commands=160,main_seconds=1800,learning_seconds=1,model_bytes=2*2**20,
                    baseline329=[json.loads(x) for x in (OUT/mode/'cases.jsonl').read_text().splitlines() if json.loads(x)['role']=='FINAL329'])
                save(folder/'protocol.json',gp);save(folder/'launch.json',dict(epoch=time.time(),model=candidate['model']));save(OUT/'status.json',dict(state='FRESH16_THEN_INDEPENDENT_RELOAD78',epoch=time.time()))
                call('gatemain',seconds=1803);gate=read(folder/'summary.json');stopped(folder);assert gate['state']=='COMPLETE'
                physics+=gate['physics'];commands+=gate['commands'];elapsed+=gate['elapsed_s']
                call('gateaudit',seconds=600);checked=read(folder/'audit.json');assert checked['state']=='PASS';gate_audit=checked;seal(folder);completed.append('gate')
                reason='SIX_NATIVE_GATES_PASS_PENDING_BACKUP_AND_USER_REVIEW' if gate['native_gate'] else 'FRESH_OR_RELOAD_GATE_STOP_NO_ADAPTATION';break
    except BaseException:
        audited=False;reason='EVIDENCE_OR_CONTRACT_FAILURE_STOP_PRESERVE_ALL'
        save(DOC/'T474_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_restart=True,raw_preserved=True))
        # Recompute exact completed summaries, never substitute missing counts0.
        summaries=[];missing=False
        for name in ['verified','joint','priority','gate']:
            folder=OUT/name
            if folder.exists():
                if (folder/'summary.json').exists():summaries.append(read(folder/'summary.json'))
                else:missing=True
        physics=None if missing else sum(v['physics'] for v in summaries)
        commands=None if missing else p['preflight_commands']+sum(v['commands'] for v in summaries)
        elapsed=None if missing else sum(v['elapsed_s'] for v in summaries)
    ids=all_stopped()
    if physics is not None:
        assert physics<=p['max_physics'] and commands<=p['max_commands'] and elapsed<=p['max_main_seconds']
        assert physics+p['night_prior_physics']<=p['night_max_physics'] and commands+p['night_prior_commands']<=p['night_max_commands']
    report=dict(state='COMPLETE_AUDITED_LOCAL' if audited else 'STOPPED_RAW_PRESERVED',reason=reason,completed=completed,ledger=ledger,verification=verification,current=current,gate=gate,gate_audit=gate_audit,
        physics=physics,commands=commands,main_elapsed_s=elapsed,night_physics=p['night_prior_physics']+physics if physics is not None else None,night_commands=p['night_prior_commands']+commands if commands is not None else None,
        remote='PENDING',six_gate=False,eight_started=False,user_review_required_before8=True,all_scientific_processes_stopped=True)
    save(OUT/'summary.json',dict(state='SEALED_FINITE_NIGHT_ROOT',pid=os.getpid(),physics=physics,commands=commands,main_elapsed_s=elapsed,completed=completed,reason=reason))
    save(OUT/'report_before_backup.json',report);save(DOC/'T474_FINAL_REPORT.json',report);save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),finalizer_writes_only_docs_after_this_point=True))
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock']]
    save(DOC/'T474_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
    try:
        c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T474'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-180))
        if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-6000:])
        remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T474.json')['state']
    except BaseException:save(DOC/'T474_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
    report['remote']=remote;report['six_gate']=bool(audited and gate and gate.get('native_gate') and gate_audit and gate_audit['state']=='PASS' and remote=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
    if report['six_gate']:report['state']='WAIT_USER_REVIEW_BEFORE_8M'
    save(DOC/'T474_FINAL_REPORT.json',report);save(DOC/'T474_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,six_gate=report['six_gate'],remote=remote,eight_started=False))
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-10 T474冻结控制复验与联合学习夜间接力结束，completed={completed}/reason={reason}/physics={physics}/commands={commands}/remote={remote}/six_gate={report['six_gate']}；两固定分支/单Actor部署，T473失败原件保留；无8，用户看后再决定，Phase隔离。详overnight_20261110/T474_FINAL_REPORT。\n\n"+f.read_text())
    subprocess.run(['git','add','stage/docs/overnight_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    subprocess.run(['git','commit','-m','Report verified offline Stage night relay and actual independent gates'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
        if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
    except BaseException:save(DOC/'T474_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))

if __name__=='__main__':
    try:main()
    except BaseException:save(DOC/'T474_FINALIZER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_scientific_restart=True))
