"""Two bounded daytime arms, complete admission and single finalizer."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t555_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def admission(p,ledger,remaining_rounds,now,used_bytes):
    rp=2*sum(j['case']['step_cap'] for j in p['training_jobs']+p['evaluation_jobs']);fp=sum(j['case']['step_cap'] for j in p['final_jobs'])
    science_seconds=remaining_rounds*(p['main_seconds']+p['audit_seconds'])+p['final_reserve_seconds']
    return bool(now+science_seconds+30<p['absolute_native_stop'] and now+science_seconds+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop'] and now+remaining_rounds*(p['main_seconds']+p['audit_seconds'])<p['absolute_learning_stop'] and ledger['physics']+remaining_rounds*rp+fp<=p['max_physics'] and ledger['commands']+remaining_rounds*88+411<=p['max_commands'] and ledger['main_seconds']+remaining_rounds*p['main_seconds']+p['final_reserve_seconds']<=p['max_main_seconds'] and used_bytes+remaining_rounds*p['child_reserve_bytes']+p['final_bytes']<=p['max_bytes'])

def main():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'launch.json').exists()
    assert read(OUT/'native_contract/audit.json')['state']=='PASS';R.G.stopped(OUT/'native_contract')
    lock=(OUT/'driver.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()==source['root_tree']
    start=time.time();end=p['absolute_all_stop'];e=read(OUT/'native_contract/summary.json')
    ledger=dict(state='RUNNING',physics=e['physics'],commands=e['commands'],main_seconds=e['elapsed_s'],completed=['native_contract'],scientific_pids=[e['pid'],read(OUT/'native_contract/audit.json')['pid']],live_Adam=0,shadow_Adam=0,retained=0,epoch=start)
    assert admission(p,ledger,2,start,R.used(OUT)),'Entire two-arm plus final worst reserve unavailable; no training started'
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,all_stop=end,native_stop=p['absolute_native_stop'],learning_stop=p['absolute_learning_stop'],source_remote=source['remote_commit'],root_tree=source['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,offline_no_network_or_LLM_for_science=True))
    arms={};final=None;failure=None
    def status(state,**kw):
        ledger.update(state=state,epoch=time.time(),**kw);save(OUT/'LEDGER.json',ledger);save(DOC/'STATUS.json',dict(driver_pid=os.getpid(),arms=arms,ledger=ledger,epoch=time.time()))
    def manifest(folder):
        files=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.pyc'])
        rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files]
        save(folder/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=rows,epoch=time.time()));assert all(sha(ROOT/x['path'])==x['sha256'] for x in rows)
    def run(action,child,cap):
        assert time.time()+cap+10<end;folder=OUT/child
        with (folder/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t555_runtime.py'),action,child],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ledger['scientific_pids'].append(c.pid);status('RUNNING_'+action,child=child,active_pid=c.pid)
            try:c.wait(timeout=cap);assert c.returncode==0,(action,child,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
        assert not Path('/proc',str(c.pid)).exists()
    def execute(child,c):
        folder=OUT/child;assert not folder.exists();folder.mkdir();c['_root_sha256']=sha(OUT/'protocol.json');save(folder/'protocol.json',c)
        run('main',child,c['main_seconds']);ledger['scientific_pids'].extend(R.G.stopped(folder));s=read(folder/'summary.json')
        ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s']
        assert ledger['physics']<=p['max_physics'] and ledger['commands']<=p['max_commands'] and ledger['main_seconds']<=p['max_main_seconds']
        run('audit',child,p['audit_seconds']);assert read(folder/'audit.json')['state']=='PASS';manifest(folder);ledger['completed'].append(child);return s
    try:
        manifest(OUT/'native_contract');rp=2*sum(j['case']['step_cap'] for j in p['training_jobs']+p['evaluation_jobs'])
        for i,arm in enumerate(p['arms']):
            assert admission(p,ledger,2-i,time.time(),R.used(OUT)),'Complete remaining reserve unavailable'
            s=execute('round_00_'+arm,dict(kind='round',arm=arm,round=0,initial=p['initial'],before_check=p['before_check'],training_seeds=p['training_seeds_fixed'],max_physics=rp,max_commands=88,main_seconds=p['main_seconds'],learning_seconds=p['learning_seconds'],child_bytes=p['child_reserve_bytes']))
            r=s['result'];arms[arm]=dict(accepted=r['accepted'],model=r['model'],check=r['check'],proposal=r['proposal']);ledger['live_Adam']+=r['live_Adam'];ledger['shadow_Adam']+=r['shadow_Adam'];ledger['retained']+=int(r['accepted']);status('ARM_COMPLETE',arm=arm)
        eligible=[a for a in p['arms'] if arms[a]['accepted']]
        if eligible:
            selected=sorted(eligible,key=lambda a:(-len(arms[a]['check']['success_ids']),arms[a]['check']['distance_sum'],p['arms'].index(a)))[0]
            assert admission(p,ledger,0,time.time(),R.used(OUT))
            fp=sum(j['case']['step_cap'] for j in p['final_jobs'])
            s=execute('final407',dict(kind='final',initial=arms[selected]['model'],max_physics=fp,max_commands=411,main_seconds=p['final_reserve_seconds'],learning_seconds=0,child_bytes=p['final_bytes']))
            final=s['result'];final['selected_arm']=selected
            previous=R.G.rows(ROOT/'stage/runs/t552_joint_robust_night/final407');current=R.G.rows(OUT/'final407')
            old={r['job_id'] for r in previous if r['success'] and not r['collision']};new={r['job_id'] for r in current if r['success'] and not r['collision']}
            final.update(lost_development_ids=sorted(old-new),gained_development_ids=sorted(new-old),development_retained=old<=new and len(new)>len(old),not_stability_or_unseen_test=True)
        else:
            old=ROOT/'stage/runs/t552_joint_robust_night/final407/summary.json';s=read(old)['result']
            assert s['model']['sha256']==p['initial']['sha256'] and sha(p['initial']['path'])==p['initial']['sha256']
            final=dict(s,reused_unchanged_baseline=True,source_summary=str(old),source_sha256=sha(old),new_final_native=0)
    except BaseException:
        failure=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=failure,epoch=time.time(),raw_preserved=True,no_training_restart=True))
        # Preserve discrepancy if an interrupted child has partial accounting.
        child=ledger.get('child');f=OUT/child/'summary.json' if child else None
        if f and f.exists() and child not in ledger['completed']:ledger['partial_child_summary']=read(f)
    finally:
        gone=all(not Path('/proc',str(i)).exists() for i in ledger['scientific_pids'])
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ledger['scientific_pids'],finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        status('COMPLETE' if failure is None else 'FAILED_STOPPED')
        report=dict(state=ledger['state'],error=failure,ledger=ledger,arms=arms,final407=final,all_scientific_pids_stopped=gone,remote='PENDING',fresh16=None,reload78=None,globally_promoted=False,epoch=time.time())
        save(DOC/'FINAL_REPORT.json',report)
        (DOC/'FINAL_REPORT.md').write_text('# T555 真实结果\n\n见FINAL_REPORT.json。局部同状态等步长对照不等于全序列相同步长；保持代理非闭环保证。未执行验收为null，未变化模型复用既有407。\n')
        if gone and time.time()+p['transport_seconds']+p['source_seconds']+15<end:
            try:
                c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T555'],cwd=ROOT,capture_output=True,timeout=p['transport_seconds']);assert c.returncode==0,c.stdout.decode(errors='replace')[-1500:]+c.stderr.decode(errors='replace')[-1500:]
                report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T555.json')['state']
            except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
        save(DOC/'FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-15 T555有限两臂已结束，'+report['state']+'；逐ID和完整407见day_20261115/t555/FINAL_REPORT；remote='+report['remote']+'。无泛化/晋升声明，不重启。\n\n'+f.read_text())
        try:
            subprocess.run(['git','add','stage/docs/day_20261115/t555','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True)
            subprocess.run(['git','commit','-m','Report T555 actual-displacement controlled pilot'],cwd=ROOT,check=True,capture_output=True)
            if time.time()+p['source_seconds']+5<end:
                c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,capture_output=True,timeout=p['source_seconds']);assert c.returncode==0
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
