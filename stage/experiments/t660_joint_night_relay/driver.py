"""Finite offline relay, explicit full-block admission and unconditional reports."""
from pathlib import Path
import json,hashlib,os,signal,subprocess,sys,time,traceback
import admission
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t660_joint_night_relay';DOC=ROOT/'stage/docs/overnight_20261119'
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'));import control as Resource

def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):
    Path(f).parent.mkdir(parents=True,exist_ok=True);Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def used(f):return sum(x.stat().st_size for x in Path(f).rglob('*') if x.is_file())

def call(action,name,seconds,children):
    with (OUT/name/(action+'.log')).open('xb') as log:
        c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t660_runtime.py'),action,name],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        children.append(c.pid)
        save(OUT/'status.json',dict(state='ACTIVE',child=name,action=action,child_pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
        try:c.wait(timeout=seconds);assert c.returncode==0,(action,name,c.returncode)
        except BaseException:
            try:os.killpg(c.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:c.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            raise

def main():
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    Resource.resources();os.sched_setaffinity(0,{14})
    save(OUT/'DRIVER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],science_network=False))
    log=(OUT/'affinity.log').open('xb');helper=subprocess.Popen(['/usr/bin/python3',str(HERE/'affinity_guard.py')],cwd=ROOT,stdout=log,stderr=log)
    children=[];results=[];error=None;best=None
    ledger=dict(physics=0,commands=0,main_seconds=0.,Actor_live=0,Value_live=0,children=[],denied=[])
    def child(name,template,**kw):
        nonlocal best
        Resource.resources();assert helper.poll() is None
        c=dict(p);c.update(template);c.update(kw);c['out']=str(OUT/name);c['arm']=kw.get('arm',template['arm'])
        learning=c['kind']=='learn';now=time.time()
        okay,checks=admission.allowed(p,c,ledger,now,used(OUT))
        if not okay:
            ledger['denied'].append(dict(name=name,reason='FULL_WORST_TIME_COMPUTE_STORAGE_ADMISSION_DENIED',checks=checks,epoch=now));save(OUT/'LEDGER.json',ledger);return None
        folder=OUT/name;folder.mkdir();(folder/'episodes').mkdir();(DOC/name).mkdir()
        c['_root_sha256']=sha(OUT/'protocol.json');save(folder/'protocol.json',{k:v for k,v in c.items() if k not in p or p[k]!=v})
        started=time.time();science_end=None;summary=None;audit=None
        try:
            call('main' if learning else 'frozen_main',name,c['main_seconds'],children);science_end=time.time()
            summary=read(folder/'summary.json');call('audit' if learning else 'frozen_audit',name,c['audit_seconds'],children)
            audit=read(folder/'audit.json');assert audit['state']=='PASS'
        finally:
            if summary is None and (folder/'summary.json').exists():summary=read(folder/'summary.json')
            actual=dict(name=name,kind=c['kind'],state=summary['state'] if summary else 'FAILED_NO_SUMMARY',audit=audit,
                physics=summary['physics'] if summary else c['max_physics'],commands=summary['commands'] if summary else c['max_commands'],
                main_seconds=(science_end or time.time())-started,
                Actor_live=summary.get('live_Actor',0) if summary else c['actor_cap'],
                Value_live=summary.get('live_Value',0) if summary else c['value_cap'],counts_exact=summary is not None)
            results.append(actual)
            for key in ['physics','commands','main_seconds','Actor_live','Value_live']:ledger[key]+=actual[key]
            ledger['children'].append(actual);save(OUT/'LEDGER.json',ledger)
            assert ledger['main_seconds']<=p['night_main_seconds']
        assert summary['state']=='COMPLETE'
        for generation in read(folder/'worker_generations.json') if (folder/'worker_generations.json').exists() else []:
            ids=[generation['parent'],*generation['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids);children.extend(ids)
        assert ledger['physics']<=p['night_max_physics'] and ledger['commands']<=p['night_max_commands']
        assert ledger['Actor_live']<=p['Actor_each_cap'] and ledger['Value_live']<=p['Value_each_cap']
        save(folder/'local_manifest.json',dict(files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json']))
        if learning and summary.get('best'):
            b=summary['best']
            if best is None or b['full407']['success']>best['full407']['success']:best=b
        return summary
    try:
        until=time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time()<until:
            assert helper.poll() is None;time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid']==helper.pid
        # Every science branch is independent from original T501. Scientific
        # rejection advances this list; contracts/errors abort the root.
        for index,arm in enumerate(p['arms']):
            template=p['bc_template'] if arm.startswith('bc') else p['rl_template']
            child(arm,template,arm=arm,arm_index=index)
        # Full original-start profiles are deterministic development measures.
        models=[('baseline',p['initial'])]+([] if best is None else [('best',best['model'])])
        for label,model in models:
            child('full407_'+label,p['profile_template'],initial=model,frozen_jobs=p['evaluation_jobs'],frozen_seeds=None,frozen_full=False)
        # Finite round-robin 19 unresolved fields, not seed/window search.
        # Fixed baseline coverage is useful even when all learning branches
        # reject; original raw successes AND failures are retained offline.
        for batch in range(p['coverage_batches']):
            jobs=[j for j in p['coverage_jobs'] for unused in range(p['coverage_replicates_per_batch'])]
            seeds=[[p['night_seed'],900,batch,i] for i in range(len(jobs))]
            if child(f'coverage_{batch:02d}',p['coverage_template'],initial=p['initial'],frozen_jobs=jobs,frozen_seeds=seeds,frozen_full=True) is None:break
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    for f in OUT.glob('*/worker_generations.json'):
        for g in read(f):children.extend([g['parent'],*g['workers']])
    ids=sorted(set(children));gone=all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ids,epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:helper.wait(timeout=5)
    except subprocess.TimeoutExpired:helper.terminate();helper.wait(timeout=3)
    log.close()
    report=dict(state='FAILED' if error else 'COMPLETE_FINITE_QUEUE_OR_BUDGET_STOP',error=error,ledger=ledger,best=best,
                all_scientific_processes_stopped=gone,remote='REMOTE_PENDING_LOCAL_OFFLINE_RAW_RETAINED',
                no_unseen_layout_or_universal_nonforgetting_claim=True,epoch=time.time())
    save(OUT/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT.json',report)
    if gone and time.time()+p['backup_seconds']+p['source_seconds']+30<p['absolute_all_stop']:
        try:
            with (OUT/'backup.log').open('xb') as log:
                c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'transport.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=p['backup_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert c.returncode==0;report['remote']='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:report['backup_error']=traceback.format_exc()
    save(OUT/'FINAL_REPORT_FINAL.json',report);save(DOC/'FINAL_REPORT.json',report)
    save(OUT/'status.json',dict(state=report['state'],all_science_stopped=gone,epoch=time.time()))
    status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-20 T660离线联合学习多支线结束：'+report['state']+'，remote='+report['remote']+
        '；真实branch/round/384保持与387目标/条件407及覆盖数据见overnight_20261119/FINAL_REPORT，缺测null，非普遍抗遗忘或真机通过。\n\n'+status.read_text())
    if gone:
        try:
            paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
            paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*T660*.json'))
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
            subprocess.run(['git','commit','-m','Report bounded offline joint learning overnight relay'],cwd=ROOT,check=True,capture_output=True)
            if time.time()+p['source_seconds']+10<p['absolute_all_stop']:
                # Execute outside generic runtime imports to avoid transport
                # module-name cache collisions in the evidence exporter.
                subprocess.run([str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'),'source_sync'],
                    cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'],best=best,ledger=ledger,remote=report['remote'])),flush=True)

if __name__=='__main__':main()
