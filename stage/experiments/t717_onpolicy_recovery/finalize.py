"""Bounded post-exit recovery qualification, receipts and no further learning."""
from pathlib import Path
import importlib.util,json,gzip,time,subprocess,sys,signal,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t717_final_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
D,O=C.DOC,C.OUT
GROUP='DAY_T717_ONPOLICY_RECOVERY_0'
def charge(commit):
    x=C.read(D/'TRANSPORT_LEDGER.json')
    if commit not in x['source_commits']:x['source_commits'].append(commit);x['charged_seconds']+=300;assert x['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',x)
def analysis():
    p=C.root_plan();fr=C.read(O/'FINAL_REPORT.json');train=C.read(O/'training/summary.json') if (O/'training/summary.json').exists() else {};ta=C.read(O/'training/audit.json') if (O/'training/audit.json').exists() else {};ev=C.read(O/'validation/summary.json') if (O/'validation/summary.json').exists() else {};ea=C.read(O/'validation/audit.json') if (O/'validation/audit.json').exists() else {}
    complete=fr['state']=='COMPLETE' and ta.get('state')=='PASS' and ea.get('state')=='PASS';results={}
    for role in ['recovery','original']:
        rows=[r for r in ev.get('fresh',[]) if r['role']==role];safe=[r['job_id'] for r in rows if C.M.N.T.Prior.safe(r)];collision=[r['job_id'] for r in rows if r['collision']];timeout=[r['job_id'] for r in rows if not r['success'] and not r['collision']];boundary=[r['job_id'] for r in rows if 'fullfield_start_pose' in r['request']['job'] and r.get('field_boundary')]
        results[role]=dict(total=len(rows),safe=len(safe) if complete else None,safe_ids=sorted(safe),collision_ids=sorted(collision),timeout_ids=sorted(timeout),boundary_ids=sorted(boundary),training_safe=len(set(safe)&set(p['training_ids'])) if complete else None,validation_safe=len(set(safe)&set(p['validation_ids'])) if complete else None)
    rc,original=results['recovery'],results['original'];gate=complete and rc['training_safe']>=10 and rc['validation_safe']>=4 and original['safe']>=12 and not rc['collision_ids'] and not original['collision_ids']
    report=dict(state=fr['state'],complete_panel=complete,results=results,acquisition_gate=gate,model=ev.get('model',train.get('model')),training_audit=ta.get('state','MISSING'),validation_audit=ea.get('state','MISSING'),actual_cost=C.read(O/'LEDGER.json'),retained='T698',integration='NOT_RUN_SEPARATE_FUTURE_AUTHORIZATION_REQUIRED',old403='NOT_MEASURED_PHASE1_ACQUISITION_ONLY',no_new_algorithm_or_generalization_claim=True,epoch=time.time())
    C.save(D/'PAIRED_RESULTS.json',report)
    text=['# T717 finite on-policy recovery acquisition','',f"State {fr['state']}; complete panel {complete}; acquisition gate {gate}.",'','| Final same model | Safe | Total | Training | Validation |','|---|---:|---:|---:|---:|']
    for name,r in results.items():text.append(f"| {name} | {r['safe']} | {r['total']} | {r['training_safe']} | {r['validation_safe']} |")
    text+=['','Three fresh suffix PPO blocks, frozen parent backbone and all20198 residual MLP parameters trainable; separate training-only Value. Complete forced B0 prefix excluded from policy/value gradients. No old403 loss/acceptance gate, no runtime teacher/router/planner. Last model fixed without validation selection. These are exposed same-scene diagnostic conditions;6ID training-heldout is not unseen-map testing. Report collision/timeout/boundary overlap and exact/audit status in PAIRED_RESULTS; missing panels are not negative scores. Old403 retention and fixed-capacity integration remain untested here, so T698 stays retained. No automatic next study or parameter/seed rescue.']
    (D/'FINDINGS.md').write_text('\n'.join(text)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T717新授权闭环恢复PPO结束：{fr['state']}，完整面板={complete}、预声明能力门={gate}；固定12训练/6验证、完整18条件恢复与18原起点见day_20261124/t717/PAIRED_RESULTS。原GRU冻结/20198头可塑，无旧403保持或自动整合，T698保持。\n\n"+f.read_text())

def backup():
    C.stopped();start=time.time();p=C.root_plan();l=C.read(D/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=900
    files=[O/f for f in ['protocol.json','LEDGER.json','FINAL_REPORT.json','SEALED.json','fit_calls.jsonl','training/summary.json','training/audit.json','training/ROUNDS.json','training/cases.jsonl.gz','validation/summary.json','validation/audit.json','validation/cases.jsonl.gz']]+[D/f for f in ['PAIRED_RESULTS.json','FINDINGS.md','FIXED_SPLIT.json','PREFLIGHT.json','PREPARATION_REJECTION.json']]
    model=C.read(D/'PAIRED_RESULTS.json').get('model')
    if model:files.append(Path(model['path']))
    files=[f for f in files if f.exists()];hashes={str(f.relative_to(ROOT)):C.sha(f) for f in files};size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    C.save(D/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),bytes=size,full_histories_local_pending=True,model_included=bool(model)))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T717 primary180')));signal.alarm(175)
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
    paths=['stage/docs/day_20261124/t717','stage/experiments/t717_onpolicy_recovery','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']+[str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],check=True,cwd=ROOT);dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,dirty
    subprocess.run(['git','commit','-m','Record bounded T717 PPO acquisition including failures'],check=True,cwd=ROOT);target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
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
