"""Pre-registered offline finite command-credit relay, no LLM/network needed.

09:00 hard stop; no failed-directory restart, grid or unregistered fallback.
"""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t510_overnight_command_credit';DOC=ROOT/'stage/docs/overnight_20261112'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):
    p=Path(f);q=Path(str(p)+'.partial');q.write_text(json.dumps(d,ensure_ascii=False)+'\n');q.replace(p)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def stopped(folder):
    ids=[read(folder/'summary.json')['pid']]
    if (folder/'audit.json').exists():ids.append(read(folder/'audit.json')['pid'])
    for g in read(folder/'worker_generations.json'):ids.extend([g['parent'],*g['workers']])
    assert all(not Path('/proc',str(pid)).exists() for pid in ids);return ids
def call(action,arm,seconds):
    with (OUT/(arm+'_'+action+'.log')).open('xb') as log:
        c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t510_runtime.py'),action,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:c.wait(timeout=seconds);assert c.returncode==0,(arm,action,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(c.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                c.wait()
            raise
def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert not (OUT/'launch.json').exists()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),absolute_all_stop=end,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False,network_required_for_science=False))
    completed=[];skipped=[];allpids=[];results={};current=None
    def accounting():
        a=[read(f) for f in OUT.glob('*/summary.json')]
        return dict(physics=sum(s['physics'] for s in a),commands=p['preflight_commands']+sum(s['commands'] for s in a),main_seconds=sum(s['elapsed_s'] for s in a),accounting_lower_bound_if_child_summary_missing=any(not (OUT/k/'summary.json').exists() for k in p['arms'] if (OUT/k).exists()))
    try:
        for arm in p['arms']:
            a=accounting();reason=None
            if used(OUT)+p['arm_bytes']>p['max_bytes']:reason='ROOT_CAPACITY_FULL_WORST_CASE_NOT_AVAILABLE'
            elif used(ROOT/'stage/runs')+p['arm_bytes']>2**30:reason='GLOBAL_CAPACITY_FULL_WORST_CASE_NOT_AVAILABLE'
            elif time.time()+p['arm_main_seconds']+p['arm_audit_seconds']+p['backup_reserve_seconds']>end:reason='FULL_ARM_AND_FINALIZATION_TIME_UNAVAILABLE'
            elif time.time()+p['arm_main_seconds']>p['absolute_native_stop']:reason='NATIVE_DEADLINE_ADMISSION'
            elif time.time()+p['arm_learning_seconds']>p['absolute_learning_stop']:reason='LEARNING_DEADLINE_ADMISSION'
            elif a['physics']+p['arm_physics']>p['root_max_physics'] or a['commands']+81>p['root_max_commands'] or a['main_seconds']+p['arm_main_seconds']>p['root_max_main_seconds']:reason='NIGHT_SHARED_BUDGET_ADMISSION'
            if reason:
                skipped.append(dict(arm=arm,reason=reason,epoch=time.time()));save(DOC/'NIGHT_LEDGER.json',dict(completed=completed,skipped=skipped,accounting=accounting()));continue
            current=arm;folder=OUT/arm;folder.mkdir();save(folder/'protocol.json',dict(_root_sha256=sha(OUT/'protocol.json'),phase=arm,arm=arm,out=str(folder),max_bytes=p['arm_bytes'],max_physics=p['arm_physics'],max_commands=81,main_seconds=p['arm_main_seconds'],learning_seconds=p['arm_learning_seconds']))
            save(OUT/'status.json',dict(state='RUNNING_'+arm,epoch=time.time(),completed=completed,skipped=skipped,accounting=accounting()));save(DOC/'NIGHT_STATUS.json',dict(state='RUNNING_'+arm,epoch=time.time(),no_network_needed=True))
            call('main',arm,p['arm_main_seconds']+3);assert read(folder/'summary.json')['state']=='COMPLETE';allpids+=stopped(folder)
            call('audit',arm,p['arm_audit_seconds']);assert read(folder/'audit.json')['state']=='PASS';allpids+=stopped(folder)
            manifest=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json'];save(folder/'local_manifest.json',dict(state='LOCAL_ALL_SHA_VERIFIED_REMOTE_PENDING',files=manifest));assert all(sha(ROOT/x['path'])==x['sha256'] for x in manifest)
            results[arm]=read(folder/'summary.json')['result'];completed.append(arm);current=None
            save(DOC/'NIGHT_LEDGER.json',dict(completed=completed,skipped=skipped,accounting=accounting(),results={k:{x:v for x,v in z.items() if x!='update'} for k,z in results.items()}))
        assert all(not Path('/proc',str(pid)).exists() for pid in allpids)
        accepted=[(i,k,z) for i,(k,z) in enumerate(results.items()) if z['accepted']]
        accepted.sort(key=lambda x:(-x[2]['field_qualified_goals'],x[2]['field24']['contact'],x[0]))
        best=accepted[0][2]['model'] if accepted else p['initial'];reason='REGISTERED_FINITE_ARMS_COMPLETE' if not skipped else 'FINITE_ARMS_WITH_EXPLICIT_BUDGET_SKIPS'
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED',reason=reason,epoch=time.time()));save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(allpids)),finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        remote='PENDING'
        try:
            remain=min(900,end-time.time()-180);assert remain>40
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T510'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remain)
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T510.json')['state']
        except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),science_not_blocked_by_network=True,raw_preserved=True))
        report=dict(state='SCIENTIFIC_COMPLETE_ALL_EXECUTED_AUDITS_PASS',reason=reason,completed=completed,skipped=skipped,arms=results,selected_model=best,selected_by_development_not_holdout=True,accounting=accounting(),remote=remote,all_scientific_processes_stopped=True,old275=None,six54=None,eight54=None,fresh16=None,all_field_stable=any(z.get('field_qualified_goals')==24 for z in results.values()),epoch=time.time())
        save(DOC/'FINAL_REPORT.json',report);save(DOC/'NIGHT_STATUS.json',dict(state='FINISHED',reason=reason,epoch=time.time(),no_more_automatic_experiments=True))
        text=['# Stage 11月12日通宵固定接力结果','',reason,'','|分支|旧8|field24|接受|','|---|---|---|---|']
        for k,z in results.items():text.append('|'+k+'|'+str(z['old8'])+'|'+str(z['field24'])+'|'+str(z['accepted'])+'|')
        text+=['','缺测为null，不作为失败率；未执行分支：'+str(skipped),'','远端备份：'+remote,'','有限预注册机制完成后没有重复耗时训练。'];(DOC/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-12/13 T510离线固定三分支结束：'+reason+'，completed='+str(completed)+'，remote='+remote+'；逐臂真实guard/field与缺测见overnight_20261112/FINAL_REPORT，原接受11模型保留；无新PG/Value，未验old275/six/eight/fresh。Phase隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/overnight_20261112','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','commit','-m','Report finite offline Stage actuator-credit night'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(90,end-time.time()-5));assert c.returncode==0,c.stdout.decode(errors='replace')[-1600:]
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'NIGHT_ERROR.json',dict(error=traceback.format_exc(),current_arm=current,completed=completed,accounting=accounting(),raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
