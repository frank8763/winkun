"""Bounded post-exit recovery qualification, receipts and no further learning."""
from pathlib import Path
import importlib.util,json,gzip,time,subprocess,sys,signal,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t716_final_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
D,O=C.DOC,C.OUT
GROUP='DAY_T716_RECOVERY_0'
def charge(commit):
    x=C.read(D/'TRANSPORT_LEDGER.json')
    if commit not in x['source_commits']:x['source_commits'].append(commit);x['charged_seconds']+=300;assert x['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',x)
def analysis():
    p=C.root_plan();results={};complete=True;safe=[]
    for arm in ['compact','replay50']:
        folder=O/('recovery_'+arm);sm=C.read(folder/'summary.json') if (folder/'summary.json').exists() else {};au=C.read(folder/'audit.json') if (folder/'audit.json').exists() else {}
        valid=sm.get('state')=='COMPLETE' and au.get('state')=='PASS';complete &=valid
        rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (folder/'cases.jsonl.gz').exists() else []
        fresh=[r for r in rows if not r.get('exact')]
        if valid:
            assert {r['job_id'] for r in fresh}=={j['id'] for j in p['recovery_groups'][arm]};safe+=sm['success_ids']
        results[arm]=dict(state=sm.get('state','MISSING'),audit=au.get('state','MISSING'),safe=sm.get('success'),total=len(p['recovery_groups'][arm]),success_ids=sm.get('success_ids'),collision_ids=sm.get('collision_ids'),timeout_ids=sm.get('timeout_ids'),boundary_ids=sm.get('boundary_ids'),conditional_recovery_rows=[dict(id=r['job_id'],safe=C.M.N.T.Prior.safe(r),first=r['request']['job']['recovery_first'],remaining_m=r['remaining_m']) for r in fresh],actual_cost=sm.get('elapsed_seconds'))
    gate=complete and len(safe)==18
    report=dict(state=C.read(O/'FINAL_REPORT.json')['state'],complete_panel=complete,results=results,qualified_ids=sorted(safe),all18_recovery_qualified=gate,integration='NOT_ADMITTED_LEARNING17_DEADLINE_PASSED',new_learning=0,not_original_start_single_student_success=True,retained='T698',cost=C.read(O/'LEDGER.json'),epoch=time.time())
    C.save(D/'RECOVERY_RESULTS.json',report)
    lines=['# T716 fixed neural-teacher recovery qualification','',f"State {report['state']}, complete panel {complete}, qualified conditional recoveries {len(safe)}/18, all18 gate {gate}.",'','| Teacher | Audit | Safe | Fixed total |','|---|---|---:|---:|']
    for n,r in results.items():lines.append(f"| {n} | {r['audit']} | {r['safe']} | {r['total']} |")
    lines+=['','These are training-only teacher continuations after forced real B0 prefixes, not a single student policy succeeding from original starts. No runtime route/teacher/TaskID deployment, no traditional navigation, no new learning. Every original horizon includes the prefix. Any failed/missing recovery stops this fixed all18 correction-set proposal; no switching time/teacher/seed rescue or successful-subset fit. T668/T709 counterevidence unchanged. Integration not admitted because original17:00 learning stop passed; T698 retained. All IDs/collisions/timeouts/boundary overlap, prefix points, costs and missing audits retained in RECOVERY_RESULTS.json. No automatic next experiment.']
    (D/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T716既有神经teacher恢复资格结束：{report['state']}，完整面板={complete}，条件恢复{len(safe)}/18、全18门={gate}。0学习，非单student原起点能力；17:00已截止不整合，T698保持，见day_20261124/t716/FINDINGS及RECOVERY_RESULTS。\n\n"+f.read_text())
def backup():
    C.stopped();start=time.time();p=C.root_plan();l=C.read(D/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=900
    files=[O/f for f in ['protocol.json','LEDGER.json','FINAL_REPORT.json','SEALED.json','fit/summary.json','fit/audit.json']]+[D/f for f in ['RECOVERY_RESULTS.json','FINDINGS.md','FIXED_ASSIGNMENT.json','PREFLIGHT.json']]
    for arm in ['compact','replay50']:files +=[O/('recovery_'+arm)/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    files=[f for f in files if f.exists()];hashes={str(f.relative_to(ROOT)):C.sha(f) for f in files};size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    C.save(D/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),bytes=size,full_histories_local_pending=True,no_new_models=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T716 primary180')));signal.alarm(175)
    try:
        sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
        G.stopped=G.Client.A.stopped=C.stopped;client=G.Client.client();index=C.read(G.DOC/'RESTORE_INDEX.json');assert not any(x['group']==GROUP for x in index['groups'])
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'));result=G.Client.A.archive(GROUP,files,release,min(start+170,p['absolute_all_stop']-330),client)
            assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';C.stopped();manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json';receipt=manifest.with_name('release_receipt_'+GROUP+'.json');members=C.read(manifest)['files'];assert {r['path']:r['sha256'] for r in members}==hashes and all(C.sha(ROOT/r['path'])==r['sha256'] for r in members)
            index['groups'].append(dict(group=GROUP,manifest=str(manifest),receipt=str(receipt),members=members,remote_receipt=result));index.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);C.save(G.DOC/'RESTORE_INDEX.json',index);C.save(D/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,deleted=0,full_histories_local_pending=True))
    finally:
        signal.alarm(0);l=C.read(D/'TRANSPORT_LEDGER.json');l['charged_seconds']+=time.time()-start;assert l['charged_seconds']+300<=900;C.save(D/'TRANSPORT_LEDGER.json',l)
def watch():
    p=C.root_plan();C.save(D/'FINALIZER_STARTED.json',dict(pid=__import__('os').getpid(),epoch=time.time(),one_shot=True))
    while time.time()<p['absolute_all_stop']-600:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();charge(C.read(D/'TRANSPORT_PENDING.json')['remote_commit']);analysis()
    try:backup()
    except BaseException:C.save(D/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    C.stopped();assert time.time()+300<p['absolute_all_stop']
    paths=['stage/docs/day_20261124/t716','stage/experiments/t716_existing_teacher_recovery','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']+[str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],check=True,cwd=ROOT);dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,dirty
    subprocess.run(['git','commit','-m','Record bounded T716 neural recovery qualification including failures'],check=True,cwd=ROOT);target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (D/'final_source.log').open('xb') as f:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=f,stderr=f)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        receipt=C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target;charge(receipt['remote_commit']);C.save(D/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_VERIFIED',remote_commit=receipt['remote_commit'],root_tree=target))
    else:
        l=C.read(D/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;assert l['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',l);C.save(D/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_PRESERVED',returncode=code))
    C.save(D/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_next_science=True))
if __name__=='__main__':
    try:C.sync(final=True) if sys.argv[1]=='source' else watch()
    except BaseException:C.save(D/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,epoch=time.time()));raise
