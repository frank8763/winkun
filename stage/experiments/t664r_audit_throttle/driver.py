"""Finite day relay; device-consistent audits and full resource admission."""
from pathlib import Path
import json,hashlib,os,signal,subprocess,sys,time,traceback
import t664r_admission as admission
import t664r_wait as Wait
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t664r_audit_throttle';DOC=ROOT/'stage/docs/day_20261120/t664r'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):
    Path(f).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(f)+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(f)
def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file())
def gpu_idle():
    try:result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    except (OSError,subprocess.TimeoutExpired):return False
    return result.returncode==0 and not result.stdout.strip()
def select_device():return 'cuda' if gpu_idle() else 'cpu'
def prefix_plan(p):
    c=dict(p);c.update(p['prefix_template']);c['out']=str(OUT/'low_prefix_audit');c['arm']='ppo_binary_s005_prefix_audit'
    c['raw_sigma']=[.05,.05];c['objective_kind']='binary';c['learner_device']='cpu'
    return c
def call(action,name,seconds,children):
    assert not (OUT/name/(action+'.log')).exists()
    with (OUT/name/(action+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t664r_runtime.py'),action,name],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        children.append(c.pid)
        save(OUT/'status.json',dict(state='ACTIVE',child=name,action=action,child_pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
        try:
            c.wait(timeout=seconds);assert c.returncode==0,(action,name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise

def optimizer_counts(folder):
    rows=[json.loads(x) for x in (folder/'optimizer_calls.jsonl').read_text().splitlines()] if (folder/'optimizer_calls.jsonl').exists() else []
    return {who:{kind:sum(r[:1]==[who] and r[2:4]==[kind,'BEFORE'] for r in rows) for kind in ['actor','value']} for who in ['live','shadow','independent_history']}

def main():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Wait.resources();os.sched_setaffinity(0,{14})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log)
    common_device=p['learner_device']
    save(OUT/'LEARNER_DEVICE.json',dict(device=common_device,shared_by_all_four_arms=True,epoch=time.time()))
    children=[];results=[];pending=[];error=None;best=None
    ledger=dict(physics=0,commands=0,main_seconds=0.,audit_seconds=0.,Actor_live=0,Value_live=0,Actor_shadow=0,Value_shadow=0,Actor_independent=0,Value_independent=0,children=[],denied=[])
    def account():
        for key in ['physics','commands','main_seconds','audit_seconds','Actor_live','Value_live','Actor_shadow','Value_shadow','Actor_independent','Value_independent']:ledger[key]=sum(v[key] for v in results)
        ledger['children']=results;save(OUT/'LEDGER.json',ledger)
        assert ledger['physics']<=p['night_max_physics'] and ledger['commands']<=p['night_max_commands']
        assert ledger['main_seconds']<=p['night_main_seconds']
        assert ledger['audit_seconds']<=p['audit_total_seconds']
        for who in ['live','shadow','independent']:
            assert ledger['Actor_'+who]<=p['Actor_each_cap'] and ledger['Value_'+who]<=p['Value_each_cap']
    def finish_audit(name,c,item):
        nonlocal best
        if c['learner_device']=='cuda' and not gpu_idle():return False
        if time.time()+c['audit_seconds']+p['finalization_seconds']>=p['absolute_all_stop']:return False
        if ledger['audit_seconds']+c['audit_seconds']+5>p['audit_total_seconds']:return False
        started=time.time()
        try:call('audit' if c['kind']=='learn' else 'frozen_audit',name,c['audit_seconds'],children)
        finally:
            item['audit_seconds']+=time.time()-started
            counts=optimizer_counts(OUT/name)
            item['Actor_independent']=counts['independent_history']['actor'];item['Value_independent']=counts['independent_history']['value']
            account()
        a=read(OUT/name/'audit.json');assert a['state']=='PASS';item['audit']=a
        counts=optimizer_counts(OUT/name)
        item['Actor_independent']=counts['independent_history']['actor'];item['Value_independent']=counts['independent_history']['value']
        summary=read(OUT/name/'summary.json')
        if c['kind']=='learn' and summary.get('best'):
            b=summary['best'];b['model']['raw_sigma']=c['raw_sigma']
            if best is None or b['full407']['success']>best['full407']['success']:best=b
        account();return True
    def child(name,template,**kw):
        Wait.resources();assert helper.poll() is None
        c=dict(p);c.update(template);c.update(kw);c['out']=str(OUT/name)
        c['learner_device']=common_device if c['kind']=='learn' else 'cpu'
        okay,checks=admission.allowed(p,c,ledger,time.time(),used(OUT))
        if not okay:
            ledger['denied'].append(dict(name=name,checks=checks,epoch=time.time()));save(OUT/'LEDGER.json',ledger);return None
        folder=OUT/name;folder.mkdir();(folder/'episodes').mkdir();(DOC/name).mkdir()
        c['_root_sha256']=sha(OUT/'protocol.json');save(folder/'protocol.json',{k:v for k,v in c.items() if k not in p or p[k]!=v})
        started=time.time();summary=None;failure=None
        try:
            call('main' if c['kind']=='learn' else 'frozen_main',name,c['main_seconds'],children)
            summary=read(folder/'summary.json');assert summary['state']=='COMPLETE'
        except BaseException:failure=traceback.format_exc()
        elapsed=time.time()-started
        if summary is None and (folder/'summary.json').exists():summary=read(folder/'summary.json')
        counts=optimizer_counts(folder)
        item=dict(name=name,kind=c['kind'],state=summary['state'] if summary else 'FAILED_NO_SUMMARY',audit=None,device=c['learner_device'],
            physics=summary['physics'] if summary else c['max_physics'],commands=summary['commands'] if summary else c['max_commands'],main_seconds=elapsed,
            Actor_live=counts['live']['actor'],Value_live=counts['live']['value'],Actor_shadow=counts['shadow']['actor'],Value_shadow=counts['shadow']['value'],Actor_independent=0,Value_independent=0,
            counts_exact=summary is not None,error=failure,audit_seconds=0.)
        results.append(item);account()
        if (folder/'worker_generations.json').exists():
            for group in read(folder/'worker_generations.json'):
                ids=[group['parent'],*group['workers']];children.extend(ids);assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        if failure:
            text=(folder/'failure.json').read_text() if (folder/'failure.json').exists() else ''
            if 'GPU_RESOURCE_WAIT_REACHED_ORIGINAL_PHASE_DEADLINE' not in text:raise RuntimeError(failure)
            return summary
        if not finish_audit(name,c,item):
            item['audit_pending_reason']='Matching GPU unavailable or complete audit/finalization time denied';pending.append((name,c,item));account()
        return summary
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:
            assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        # Independently audit only the durable old low-noise prefix. It is
        # neither retraining live nor a repeated native measurement.
        prefix=prefix_plan(p)
        folder=Path(prefix['out']);folder.mkdir();(DOC/prefix['arm']).mkdir()
        prefix['_root_sha256']=sha(OUT/'protocol.json');save(folder/'protocol.json',{k:v for k,v in prefix.items() if k not in p or p[k]!=v})
        started=time.time();call('prefix_audit','low_prefix_audit',prefix['audit_seconds'],children)
        a=read(folder/'audit.json');assert a['state']=='PASS'
        results.append(dict(name='low_prefix_audit',kind='audit_adoption',state='COMPLETE',audit=a,device='cpu',physics=0,commands=0,main_seconds=0.,audit_seconds=time.time()-started,
            Actor_live=0,Value_live=0,Actor_shadow=0,Value_shadow=0,Actor_independent=a['independent_Actor'],Value_independent=a['independent_Value'],counts_exact=True,error=None))
        account()
        save(OUT/'status.json',dict(state='WAITING_FOR_SCHEDULED_START',driver_pid=os.getpid(),first_start_epoch=p['first_start_epoch'],epoch=time.time()))
        while time.time()<p['first_start_epoch']:
            assert helper.poll() is None and time.time()<p['absolute_learning_stop'];time.sleep(min(1,p['first_start_epoch']-time.time()))
        for config in p['arm_configurations']:
            child(config['arm'],p['resume_template'] if config.get('resume_state') else p['rl_template'],**config)
        for name,c,item in pending:
            if finish_audit(name,c,item):item.pop('audit_pending_reason',None)
        for batch in range(p['coverage_batches']):
            jobs=[j for j in p['coverage_jobs'] for unused in range(p['coverage_replicates_per_batch'])]
            model=p['initial'] if best is None else best['model']
            seeds=[[p['night_seed'],900,batch,i] for i in range(len(jobs))]
            if child(f'coverage_{batch:02d}',p['coverage_template'],initial=model,raw_sigma=model.get('raw_sigma',p['raw_sigma']),frozen_jobs=jobs,frozen_seeds=seeds,frozen_full=True) is None:break
        for name,c,item in pending:
            if item['audit'] is None and finish_audit(name,c,item):item.pop('audit_pending_reason',None)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    for f in OUT.glob('*/worker_generations.json'):
        for g in read(f):children.extend([g['parent'],*g['workers']])
    ids=sorted(set(children));gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close()
    missing=[r['name'] for r in results if r['audit'] is None]
    report=dict(state='FAILED' if error else ('COMPLETE_WITH_AUDIT_PENDING' if missing else 'COMPLETE_FINITE_QUEUE_OR_BUDGET_STOP'),error=error,ledger=ledger,best=best,audit_pending=missing,
        all_scientific_processes_stopped=gone,remote='REMOTE_PENDING_LOCAL_OFFLINE_RAW_RETAINED',epoch=time.time(),no_unseen_layout_or_universal_nonforgetting_claim=True)
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    if gone and time.time()+p['backup_seconds']+p['source_seconds']+30<p['absolute_all_stop']:
        try:
            with (OUT/'backup.log').open('xb') as log:
                c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=p['backup_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0;report['remote']='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:report['backup_error']=traceback.format_exc()
    save(OUT/'FINAL_REPORT_FINAL.json',report);save(DOC/'FINAL_REPORT.json',report)
    save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()))
    if gone:
        status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-20 T664R白天有界四臂接力结束：'+report['state']+'，remote='+report['remote']+'；实际各臂逐ID保持/场地新增/条件407/设备及审计见day_20261120/t664r/FINAL_REPORT；缺测null，不自动晋升或声称泛化。\n\n'+status.read_text())
        try:
            paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
            paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*T664R*.json'))
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
            staged=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
            if staged.returncode==1:subprocess.run(['git','commit','-m','Report bounded daytime factorial relay'],cwd=ROOT,check=True,capture_output=True)
            if time.time()+p['source_seconds']+10<p['absolute_all_stop']:
                subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'],best=best,ledger=ledger,remote=report['remote'])),flush=True)
if __name__=='__main__':main()
