"""Offline finite relay: targeted capture, mean anchor, conditional worst anchor.

Single owner; raw/audit/manifest gates between phases, never reuse a rejected
candidate as the next initial model. Network is only used after scientific exit.
"""
from pathlib import Path
import copy,hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t487_eight_only';DOC=ROOT/'stage/docs/review_20261111'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def call(action,mode,seconds):
    assert seconds>0
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t487_runtime.py'),action,mode],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    with (OUT/'subprocesses.jsonl').open('a') as f:
        f.write(json.dumps(dict(pid=c.pid,pgid=c.pid,action=action,mode=mode,epoch=time.time()))+'\n');f.flush();os.fsync(f.fileno())
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
    stopped(folder)
    files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(folder/'local_manifest.json',dict(files=files,remote='PENDING'))
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in files)

def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop'];completed=[];reason='ONE_EIGHT_ONLY_PPO_COMPLETE';remote='PENDING';allpids=[];selected=p['initial_result']
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,offline_no_LLM_network=True))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T487 harddeadline')))
    try:
        for mode in ['capture','pair']:
            folder=OUT/mode;assert not folder.exists();now=time.time()
            if mode!='capture':
                assert read(OUT/'capture/audit.json')['state']=='PASS'
                if now+2400+300+600>end or now+2400>p['absolute_native_stop'] or now+600>p['absolute_learning_stop']:
                    reason='COMPLETE_PHASE_TIME_ADMISSION_STOP';break
            folder.mkdir();pp=copy.deepcopy(p)
            if mode=='capture':
                pp.update(out=str(folder),mode=mode,max_physics=2*sum(j['case']['step_cap'] for j in p['training_jobs']),max_commands=12,main_seconds=600,learning_seconds=1)
            else:
                spent=sum(read(f)['physics'] for f in OUT.glob('*/summary.json'))
                pp.update(out=str(folder),mode=mode,anchor_mode=mode,max_physics=p['root_max_physics']-spent,max_commands=58,main_seconds=2400,learning_seconds=600)
            (folder/'protocol.json').write_text(json.dumps(pp,ensure_ascii=False,separators=(',',':'))+'\n');save(folder/'launch.json',dict(epoch=now,mode=mode,initial=p['initial']))
            save(OUT/'status.json',dict(state='RUNNING_'+mode,epoch=now,completed=completed,all_stop=end))
            try:
                call('main',mode,min(pp['main_seconds']+3,p['absolute_native_stop']-time.time()-5))
                s=read(folder/'summary.json');allpids+=stopped(folder);assert s['state']=='COMPLETE'
                call('audit',mode,min(180 if mode=='capture' else 300,end-time.time()-600))
                assert read(folder/'audit.json')['state']=='PASS';allpids+=stopped(folder);manifest(folder);completed.append(mode)
                save(DOC/'T487_LEDGER.json',dict(completed=completed,latest=s,epoch=time.time(),remote='PENDING'))
                if mode!='capture' and s['result']['chosen_arm'] is not None:
                    selected=s['result']['selected'];reason='RETAINED_PROGRESS_STOP_FURTHER_LEARNING';break
            except BaseException:
                save(DOC/'T487_SCIENTIFIC_ERROR.json',dict(mode=mode,error=traceback.format_exc(),no_retry=True,saved_raw_preserved=True));reason='EVIDENCE_OR_ENGINEERING_FAILURE_STOP';break
        summaries={f.parent.name:read(f) for f in OUT.glob('*/summary.json')}
        physics=sum(s['physics'] for s in summaries.values());commands=read(OUT/'preflight.json')['commands']+sum(s['commands'] for s in summaries.values())
        assert physics<=p['root_max_physics'] and commands<=p['root_max_commands']
        assert sum(s['elapsed_s'] for s in summaries.values())<=p['root_max_main_seconds']+9
        for f in OUT.glob('*/worker_generations.json'):allpids+=stopped(f.parent)
        groups={x['pgid'] for x in map(json.loads,(OUT/'subprocesses.jsonl').read_text().splitlines())}
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:assert int(stat.read_text().rsplit(')',1)[1].split()[2]) not in groups,('Scientificprocessstillalive',stat)
            except (FileNotFoundError,ProcessLookupError):pass
        complete=(completed==['capture','pair'] or reason=='RETAINED_PROGRESS_STOP_FURTHER_LEARNING')
        report=dict(state='COMPLETE_AUDITED_LOCAL' if complete else 'BOUNDED_QUEUE_STOPPED',reason=reason,completed=completed,
            physics=physics,commands=commands,summaries=summaries,selected=selected,remote='PENDING',eight_stability_gate=False,
            fresh16=None,reload78=None,all_scientific_processes_stopped=True,new_training_episodes=4 if 'capture' in completed else None,
            eight_only=True,old275=None,wide6=None)
        save(OUT/'report_before_backup.json',report);save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED_BACKUP',epoch=time.time(),reason=reason))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(allpids)),epoch=time.time(),finalizer_writes_only_docs_after_this_point=True))
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.pyc','.lock']]
        save(DOC/'T487_LOCAL_MANIFEST.json',dict(files=files,remote='PENDING'))
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T487'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(480,end-time.time()-120))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T487.json')['state']
        except BaseException:save(DOC/'T487_REMOTE_PENDING.json',dict(error=traceback.format_exc(),saved_raw_preserved=True))
        report['remote']=remote;save(DOC/'T487_FINAL_REPORT.json',report)
        (DOC/'T487_FINAL_REPORT.md').write_text('# T487实际结果\n\n'+f"State:{report['state']}; reason:{reason}; physics:{physics}; commands:{commands}; remote:{remote}\n\n"+json.dumps({k:selected.get(k) for k in ['wide8','wide6','old275']},ensure_ascii=False,indent=2)+'\n\n未执行fresh16导航留出/78独立原生模型读回，不宣称稳定八米。\n')
        save(DOC/'T487_STATUS.json',dict(state='FINISHED',epoch=time.time(),reason=reason,remote=remote,no_automatic_relaunch=True))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text(f"2026-11-11 T487仅宽八米PPO单候选结束，completed={completed}/reason={reason}/physics={physics}/commands={commands}/remote={remote}；保留八米{selected['wide8']['success']}/54，旧275/六米本轮未运行null，不称联合能力或稳定八米。详review_20261111/T487_FINAL_REPORT，Phase隔离。\n\n"+f.read_text())
        subprocess.run(['git','add','stage/docs/review_20261111','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report finite targeted-coverage evening relay'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(120,end-time.time()-5))
            if c.returncode:raise RuntimeError(c.stdout.decode(errors='replace')[-4000:])
        except BaseException:save(DOC/'T487_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:save(DOC/'T487_DRIVER_ERROR.json',dict(error=traceback.format_exc(),no_scientific_retry=True,saved_raw_preserved=True))
if __name__=='__main__':main()
