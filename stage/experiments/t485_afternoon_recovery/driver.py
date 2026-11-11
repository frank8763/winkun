"""One offline owner: old8 capture, independent audit, one rehearsal pilot, final archive."""
from pathlib import Path
import copy,json,os,signal,subprocess,time,traceback,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t485_afternoon_recovery';DOC=ROOT/'stage/docs/review_20261111'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def call(action,mode,seconds):
    assert seconds>0;c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t485_runtime.py'),action,mode],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'subprocesses.jsonl').open('a') as f:f.write(json.dumps(dict(pid=c.pid,pgid=c.pid,action=action,mode=mode,epoch=time.time()))+'\n');f.flush();os.fsync(f.fileno())
    try:
        b,_=c.communicate(timeout=seconds)
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
        if (folder/name).exists():ids.append(read(folder/name)['pid'])
    if (folder/'worker_generations.json').exists():
        for g in read(folder/'worker_generations.json'):ids+=[g['parent']]+g['workers']
    assert all(not Path('/proc',str(v)).exists() for v in ids);return ids

def manifest(folder):
    stopped(folder);files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json'];save(folder/'local_manifest.json',dict(files=files,remote='PENDING'));assert all(sha(ROOT/r['path'])==r['sha256'] for r in files)

def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];completed=[];reason='ONE_OLD_TASK_REHEARSAL_PILOT_COMPLETE';remote='PENDING';allpids=[]
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,offline_no_LLM_network=True));signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T485 harddeadline')))
    try:
        for mode in ['pair']:
            folder=OUT/mode;assert not folder.exists();now=time.time()
            if mode=='pair':
                assert read(p['capture_audit_path'])['state']=='PASS'
                if now+3600+900+600>end or now+1200>p['absolute_learning_stop']:reason='FULL_MATCHED_PAIR_TIME_ADMISSION_STOP';break
            folder.mkdir();pp=copy.deepcopy(p)
            if mode=='capture':pp.update(out=str(folder),mode=mode,max_physics=32*53334,max_commands=40,main_seconds=900,learning_seconds=1)
            else:pp.update(out=str(folder),mode=mode,max_physics=p['root_max_physics']-0,max_commands=410,main_seconds=3600,learning_seconds=1200)
            save(folder/'protocol.json',pp);save(folder/'launch.json',dict(epoch=now,mode=mode,initial=p['initial']));save(OUT/'status.json',dict(state='RUNNING_'+mode,epoch=now,completed=completed,all_stop=end))
            try:
                call('main',mode,min(pp['main_seconds']+3,p['absolute_native_stop']-time.time()-5));s=read(folder/'summary.json');allpids+=stopped(folder);assert s['state']=='COMPLETE'
                call('audit',mode,min(300 if mode=='capture' else 900,end-time.time()-600));assert read(folder/'audit.json')['state']=='PASS';allpids+=stopped(folder);manifest(folder);completed.append(mode)
                save(DOC/'T485_LEDGER.json',dict(completed=completed,latest=s,epoch=time.time(),remote='PENDING'))
            except BaseException:save(DOC/'T485_SCIENTIFIC_ERROR.json',dict(mode=mode,error=traceback.format_exc(),no_retry=True,saved_raw_preserved=True));reason='EVIDENCE_OR_ENGINEERING_FAILURE_STOP';break
        summaries={f.parent.name:read(f) for f in OUT.glob('*/summary.json')};physics=sum(s['physics'] for s in summaries.values());commands=read(OUT/'preflight.json')['commands']+sum(s['commands'] for s in summaries.values());assert physics<=p['root_max_physics'] and commands<=p['root_max_commands'];assert sum(s['elapsed_s'] for s in summaries.values())<=p['root_max_main_seconds']+6
        for f in OUT.glob('*/worker_generations.json'):allpids+=stopped(f.parent)
        groups={x['pgid'] for x in map(json.loads,(OUT/'subprocesses.jsonl').read_text().splitlines())}
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:assert int(stat.read_text().rsplit(')',1)[1].split()[2]) not in groups,('Scientificprocessstillalive',stat)
            except (FileNotFoundError,ProcessLookupError):pass
        selected=summaries.get('pair',{}).get('result',{}).get('selected',p['initial_result']);assert physics+p['previous_physics']<=12000000 and commands+p['previous_commands']<=500
        report=dict(previous_T481_physics=p['previous_physics'],previous_T481_commands=p['previous_commands'],joint_T481_T485_physics=physics+p['previous_physics'],joint_T481_T485_commands=commands+p['previous_commands'],state='COMPLETE_AUDITED_LOCAL' if completed==['pair'] else 'BOUNDED_QUEUE_STOPPED',reason=reason,completed=completed,physics=physics,commands=commands,summaries=summaries,selected=selected,paired_result=summaries.get('pair',{}).get('result'),remote='PENDING',eight_stability_gate=False,fresh16=None,reload78=None,all_scientific_processes_stopped=True)
        save(OUT/'report_before_backup.json',report);save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason));save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(allpids)),epoch=time.time(),finalizer_writes_only_docs_after_this_point=True));files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock']];save(DOC/'T485_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T485'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(600,end-time.time()-180))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T485.json')['state']
        except BaseException:save(DOC/'T485_REMOTE_PENDING.json',dict(error=traceback.format_exc(),saved_raw_preserved=True))
        report['remote']=remote;save(DOC/'T485_FINAL_REPORT.json',report);(DOC/'T485_FINAL_REPORT.md').write_text('# T485 actualresult\n\n'+f"State:{report['state']}; reason:{reason}; native:{physics}; commands:{commands}; remote:{remote}\n\n"+json.dumps({k:selected[k] for k in ['wide8','wide6','old275']},ensure_ascii=False,indent=2)+'\n\nFresh16/reload78 notallocated/notexecuted; not8stability. Allscientificprocessesstopped.\n');save(DOC/'T485_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,remote=remote,no_automatic_relaunch=True))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-11 T485接口修复复用T481真实16数据的剩余单候选学习结束，completed={completed}/reason={reason}/physics={physics}/commands={commands}/remote={remote}；选中8m{selected['wide8']['success']}/54 six{selected['wide6']['success']}/54 old{selected['old275']['success']}/275，fresh16/reload78未执行不称稳定。详review_20261111/T485_FINAL_REPORT，Phase隔离。\n\n"+f.read_text());subprocess.run(['git','add','stage/docs/review_20261111','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','commit','-m','Report bounded old-task Vtrace rehearsal pilot'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'T485_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'T485_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_scientific_retry=True,saved_raw_preserved=True))
if __name__=='__main__':main()
