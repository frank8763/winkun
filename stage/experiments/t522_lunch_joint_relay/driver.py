"""Finite autonomous teacher-validation -> student -> audit/archive relay."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t522_lunch_joint_relay';DOC=ROOT/'stage/docs/day_20261113'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):
    p=Path(f);q=Path(str(p)+'.partial');q.write_text(json.dumps(d,ensure_ascii=False)+'\n');q.replace(p)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def stopped(folder):
    ids=[read(folder/'summary.json')['pid']]
    if (folder/'audit.json').exists():ids.append(read(folder/'audit.json')['pid'])
    for g in read(folder/'worker_generations.json'):ids.extend([g['parent'],*g['workers']])
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    return ids
def manifest(folder):
    stopped(folder);rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json']
    save(folder/'local_manifest.json',dict(state='LOCAL_ALL_SHA_PASS_REMOTE_PENDING',files=rows));assert all(sha(ROOT/r['path'])==r['sha256'] for r in rows)
def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert not (OUT/'launch.json').exists()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False))
    completed=[];allpids=[];reason='BOUNDED_FULLFIELD_ONPOLICY';result=None
    def accounting():
        ss=[read(f) for f in OUT.glob('*/summary.json')]
        return dict(physics=sum(s['physics'] for s in ss),commands=p['preflight_commands']+sum(s['commands'] for s in ss),main_seconds=sum(s['elapsed_s'] for s in ss))
    def call(action,phase,seconds):
        with (OUT/(phase+'_'+action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t522_runtime.py'),action,phase],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:c.wait(timeout=seconds);assert c.returncode==0,(phase,action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(c.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    c.wait()
                raise
    def execute(phase,override,maincap,auditcap):
        nonlocal allpids
        a=accounting();assert a['main_seconds']+maincap<=p['root_max_main_seconds'] and a['physics']+override['max_physics']<=p['root_max_physics'] and a['commands']+override['max_commands']<=p['root_max_commands']
        assert time.time()+maincap+auditcap+660<end and time.time()+maincap<p['absolute_native_stop']
        folder=OUT/phase;folder.mkdir();override.update(phase=phase,out=str(folder),main_seconds=maincap,audit_seconds=auditcap,_root_sha256=sha(OUT/'protocol.json'));save(folder/'protocol.json',override)
        save(OUT/'status.json',dict(state='RUNNING_'+phase,epoch=time.time(),completed=completed,accounting=a))
        call('main',phase,maincap+3);assert read(folder/'summary.json')['state']=='COMPLETE';allpids+=stopped(folder)
        call('audit',phase,auditcap);assert read(folder/'audit.json')['state']=='PASS';allpids+=stopped(folder);manifest(folder);completed.append(phase)
        save(DOC/'T522_LEDGER.json',dict(completed=completed,accounting=accounting(),epoch=time.time()))
        return read(folder/'summary.json')['result']
    try:
        results={}
        for phase in p['registered_order']:
            limit=p['phase_limits'][phase];a=accounting()
            if time.time()+limit['main_seconds']+limit['audit_seconds']+660>=end or time.time()+limit['main_seconds']>=p['absolute_native_stop']:
                save(DOC/'T522_SKIPPED_'+phase.upper()+'.json',dict(reason='INSUFFICIENT_COMPLETE_REGISTERED_WORST_CASE_TIME',epoch=time.time()));break
            result=execute(phase,dict(limit),limit['main_seconds'],limit['audit_seconds']);results[phase]=result
            if result['accepted']:break
        reason='JOINT_OLD8_AND_FIELD12_GATE_PASSED' if result and result['accepted'] else 'FINITE_MECHANISMS_COMPLETE_OR_TIME_ADMISSION_STOP'
        save(DOC/'T522_RESULTS.json',results)
        for phase in completed:allpids+=stopped(OUT/phase)
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED',reason=reason,epoch=time.time()))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(allpids)),finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        remote='PENDING'
        try:
            remain=min(600,end-time.time()-120);assert remain>40
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T522'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remain)
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2500:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T522.json')['state']
        except BaseException:save(DOC/'T522_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE_ALL_EXECUTED_AUDITS_PASS',reason=reason,completed=completed,teacher=None,student=result,branches=results,accounting=accounting(),remote=remote,old275=None,wide6=None,fresh16=None,under8_gate=False,epoch=time.time(),all_scientific_processes_stopped=True)
        save(DOC/'T522_FINAL_REPORT.json',report);save(DOC/'T522_STATUS.json',dict(state='FINISHED',reason=reason,epoch=time.time()))
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-13 T522午间有限两机制结束，'+reason+'，completed='+str(completed)+'，remote='+remote+'；两任务新鲜联合PG/实际自主成功轨迹BC分列，原接受最佳独立保留，完整old275/six/eight未执行。详day_20261113/T522_FINAL_REPORT。Phase隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report fresh full-Actor on-policy RL and full-field behavior gate'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(120,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T522_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'T522_DRIVER_ERROR.json',dict(error=traceback.format_exc(),completed=completed,accounting=accounting(),accounting_is_lower_bound_if_child_summary_missing=True,raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
