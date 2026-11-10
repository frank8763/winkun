"""Offline finite scientific relay; network is only used after all science stops."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t463_evening_2000';DOC=ROOT/'stage/docs/review_20261110'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def call(*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),*args],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
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


def seal(folder):
    ids=scientific_stopped(folder)
    save(folder/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
    rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*'))
        if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'] and f.name!='local_manifest.json']
    save(folder/'local_manifest.json',dict(files=rows,remote='PENDING'))
    for r in rows:assert sha(ROOT/r['path'])==r['sha256']


def main():
    import run as X
    root=read(OUT/'protocol.json');end=root['absolute_all_stop']
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T46320:00hardstop')))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    current=root['initial_candidate'];ledger=[];physics=0;commands=24;elapsed=0.;gate=None;remote='PENDING'
    reason='FIXED_MECHANISMS_COMPLETE';completed=[];all_audited=True
    try:
        for mode in root['modes']:
            now=time.time();folder=OUT/mode
            maxworst=sum(j['case']['step_cap'] for j in root['science']['main_jobs']+root['science']['old275_jobs'])
            if mode=='ppo':maxworst+=2*sum(j['case']['step_cap'] for j in root['science']['ppo_training_jobs'])
            maxcmd=345+(16 if mode=='ppo' else 0)
            # Reserve a complete phase and full gate opportunity, not a partial run.
            if (now+1800+600+2400+900>=end or now+1800>root['absolute_native_stop']
                or now+900>root['absolute_learning_stop'] or physics+maxworst+4000000>root['max_physics']
                or commands+maxcmd+160>root['max_commands'] or elapsed+1800>root['max_main_seconds']):
                reason='NEXT_FIXED_PHASE_FULL_WORST_BUDGET_NOT_ADMITTED';break
            assert not folder.exists();folder.mkdir()
            p=X.child_config(root,mode,current['model'],current['success_ids'])
            p['main_seconds']=min(1800,int(root['absolute_native_stop']-now))
            p['learning_seconds']=min(900,int(root['absolute_learning_stop']-now))
            save(folder/'protocol.json',p);save(folder/'launch.json',dict(epoch=now,mode=mode,
                source_protocol_sha256=sha(OUT/'protocol.json'),input_model=current['model']))
            save(OUT/'status.json',dict(state=mode.upper()+'_RUNNING',epoch=time.time(),driver=os.getpid()))
            try:
                call('childmain',mode,seconds=p['main_seconds']+3)
                s=read(folder/'summary.json');scientific_stopped(folder)
                assert s['state']=='COMPLETE'
                call('childaudit',mode,seconds=min(600,end-time.time()-900))
                assert read(folder/'audit.json')['state']=='PASS';seal(folder)
            except BaseException:
                save(folder/'driver_failure.json',dict(error=traceback.format_exc(),no_restart=True,raw_preserved=True))
                if (folder/'summary.json').exists():
                    s=read(folder/'summary.json');physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
                    try:seal(folder)
                    except BaseException:save(folder/'seal_error.json',dict(error=traceback.format_exc()))
                all_audited=False;reason='SHARED_NUMERIC_OR_EVIDENCE_CONTRACT_STOP';completed.append(mode);break
            completed.append(mode);physics+=s['physics'];commands+=s['commands'];elapsed+=s['elapsed_s']
            candidate=s['candidate'];accepted=set(current['success_ids'])<=set(candidate['success_ids'])
            entry=dict(mode=mode,state='COMPLETE',input_model=current['model'],candidate=candidate,accepted=accepted,
                physics=s['physics'],commands=s['commands'],elapsed_s=s['elapsed_s'])
            ledger.append(entry);save(OUT/'ledger.json',ledger)
            if accepted:current=candidate
            assert physics<=root['max_physics'] and commands<=root['max_commands'] and elapsed<=root['max_main_seconds']
            if candidate['development_admitted']:
                assert accepted and len(current['success_ids'])==329
                gfolder=OUT/'gate';now=time.time()
                if now+1800+600+900>=end or now+1800>=root['absolute_native_stop']:
                    reason='DEVELOPMENT329_PASS_GATE_WORST_TIME_NOT_ADMITTED';break
                assert not gfolder.exists();gfolder.mkdir()
                p=X.child_config(root,mode,current['model'],current['success_ids'])
                baseline=[r for r in map(json.loads,(folder/'cases.jsonl').read_text().splitlines()) if r['role']=='FINAL329']
                p.update(id='T463_INDEPENDENT_SIX_METRE_GATES',out=str(gfolder),max_bytes=root['gate_partition_bytes'],
                    max_physics=4000000,max_commands=160,main_seconds=1800,learning_seconds=1,
                    model_bytes=2*2**20,baseline329=baseline)
                save(gfolder/'protocol.json',p);save(gfolder/'launch.json',dict(epoch=now,model=current['model']))
                try:
                    call('gatemain',seconds=1803);gs=read(gfolder/'summary.json');scientific_stopped(gfolder)
                    assert gs['state']=='COMPLETE'
                    call('gateaudit',seconds=min(600,end-time.time()-900));assert read(gfolder/'audit.json')['state']=='PASS'
                    gate=gs;seal(gfolder)
                except BaseException:
                    save(gfolder/'driver_failure.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));all_audited=False
                    if (gfolder/'summary.json').exists():gate=read(gfolder/'summary.json')
                    try:seal(gfolder)
                    except BaseException:pass
                completed.append('gate')
                if gate:physics+=gate['physics'];commands+=gate['commands'];elapsed+=gate['elapsed_s']
                reason='SIX_NATIVE_GATES_PASS_PENDING_BACKUP' if gate and gate['native_gate'] else 'NAVIGATION_HOLDOUT_OR_RELOAD_GATE_STOP'
                break
        for name in completed:
            if (OUT/name/'summary.json').exists():scientific_stopped(OUT/name)
        assert physics<=root['max_physics'] and commands<=root['max_commands'] and elapsed<=root['max_main_seconds']
    except BaseException:
        reason='DRIVER_ERROR_PRESERVE_ALL_RAW';all_audited=False
        save(OUT/'driver_error.json',dict(error=traceback.format_exc(),epoch=time.time(),no_restart=True))
    # All science has exited. Freeze this entire run before binary backup.
    # After SEALED is written, finalizer writes only outside OUT.
    try:
        ids=[]
        for name in completed:
            if (OUT/name/'SEALED.json').exists():ids+=read(OUT/name/'SEALED.json')['scientific_pids']
        save(OUT/'summary.json',dict(state='SEALED_FINITE_ROOT',pid=os.getpid(),completed=completed,reason=reason,
            physics=physics,commands=commands,main_elapsed_s=elapsed,all_scientific_pids=ids))
        report=dict(state='COMPLETE_AUDITED_LOCAL' if all_audited else 'STOPPED_EVIDENCE_PRESERVED',
            reason=reason,completed=completed,ledger=ledger,current=current,gate=gate,physics=physics,commands=commands,
            main_elapsed_s=elapsed,remote='PENDING',six_gate=False,eight_started=False,
            all_scientific_processes_stopped=all(not Path('/proc',str(pid)).exists() for pid in ids))
        save(OUT/'report_before_backup.json',report);save(DOC/'T463_FINAL_REPORT.json',report)
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time(),
            finalizer_writes_only_docs_after_this_point=True,driver_pid_not_a_scientific_writer=os.getpid()))
        if time.time()+30<end:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T463'],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(900,end-time.time()-120))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-5000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T463.json')['state']
        report['remote']=remote
        report['six_gate']=bool(all_audited and gate and gate.get('native_gate') and remote=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED')
        save(DOC/'T463_FINAL_REPORT.json',report);save(DOC/'T463_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,six_gate=report['six_gate']))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text(f"2026-11-10 T463晚间有限接力已结束，reason={reason}，completed={completed}，physics={physics} commands={commands} main_seconds={elapsed}；remote={remote}/six_gate={report['six_gate']}，8未实现未启动。固定A聚合/B有限支持函数界/C新鲜PPO按预算实际执行，原328/safeT255/所有失败保持，Phase隔离。详review_20261110/T463_FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/review_20261110','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report finite Stage evening relay and actual gates'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if time.time()+10<end:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
    except BaseException:
        save(DOC/'T463_FINALIZATION_ERROR.json',dict(error=traceback.format_exc(),remote=remote,raw_preserved=True,no_science_restart=True))


if __name__=='__main__':main()
