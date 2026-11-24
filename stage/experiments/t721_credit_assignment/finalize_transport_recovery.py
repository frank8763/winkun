"""Bounded post-exit recovery qualification, receipts and no further learning."""
from pathlib import Path
import importlib.util,json,gzip,time,subprocess,sys,signal,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t721_final_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
D,O=C.DOC,C.OUT
GROUP='OVERNIGHT_T721_CREDIT_COMPARISON_PRIMARY_0'
def charge(commit):
    x=C.read(D/'TRANSPORT_LEDGER.json')
    if commit not in x['source_commits']:x['source_commits'].append(commit);x['charged_seconds']+=300;assert x['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',x)
def analysis():
    p=C.root_plan();final=C.read(O/'FINAL_REPORT.json');complete=final['state']=='COMPLETE';arms={};learning=[]
    def rowlist(folder):
        f=folder/'cases.jsonl.gz'
        return [json.loads(v) for v in gzip.decompress(f.read_bytes()).decode().splitlines()] if f.exists() else []
    def panel(name):
        folder=O/name;sm=C.read(folder/'summary.json') if (folder/'summary.json').exists() else {};au=C.read(folder/'audit.json') if (folder/'audit.json').exists() else {};valid=sm.get('state')=='COMPLETE' and au.get('state')=='PASS'
        rows=[r for r in rowlist(folder) if not r.get('exact')];safe=sorted(r['job_id'] for r in rows if C.M.N.T.Prior.safe(r))
        return dict(state=sm.get('state','MISSING'),audit=au.get('state','MISSING'),verified=valid,safe=len(safe) if valid else None,total=len(p['panels'][name.split('_',1)[1]]['jobs']),success_ids=safe if valid else None,collision_ids=sm.get('collision_ids') if valid else None,timeout_ids=sm.get('timeout_ids') if valid else None,boundary_ids=sm.get('boundary_ids') if valid else None,model=sm.get('model'),_rows=rows)
    sourceids={j['id'] for j in p['recovery_jobs']};train=set(p['training_ids']);validation=set(p['validation_ids']);base={r['job_id']:r for r in p['panels']['exposed174']['baseline_rows']};bs={k for k,r in base.items() if C.M.N.T.Prior.safe(r)}
    # B0 retained403 membership is read from already audited complete T701/T699 evidence if available.
    b0loss=p['baseline_old403_lost']
    for arm in ['gae95','mc1']:
        pp={k:panel(f'{arm}_{k}') for k in p['panels']};complete &=all(v['verified'] for v in pp.values());ex=pp['exposed174'];rec=pp['recovery18'];old=pp['old403'];good=set(ex['success_ids'] or []);recover=set(rec['success_ids'] or []);oldgood=set(old['success_ids'] or [])
        acquisition=all(v['verified'] for v in [ex,rec]) and len(recover&train)>=10 and len(recover&validation)>=4 and len(good&sourceids)>=12 and not rec['collision_ids'] and not [r for r in ex['_rows'] if r['job_id'] in sourceids and r['collision']]
        practical=all(v['verified'] for v in [ex,old]) and ex['safe']>=102 and len(ex['collision_ids'])<=41 and old['total']-old['safe']<=20
        arms[arm]=dict(panels={k:{kk:vv for kk,vv in v.items() if kk!='_rows'} for k,v in pp.items()},gained_exposed=sorted(good-bs) if ex['verified'] else None,lost_exposed=sorted(bs-good) if ex['verified'] else None,original18_safe=len(good&sourceids) if ex['verified'] else None,recovery_train_safe=len(recover&train) if rec['verified'] else None,recovery_validation_safe=len(recover&validation) if rec['verified'] else None,old403_lost=sorted(set(p['required403_ids'])-oldgood) if old['verified'] else None,B0_old403_lost=b0loss,acquisition_gate=bool(acquisition),practical_gate=bool(practical),integration='NOT_RUN')
        for segment in range(3):
            folder=O/f'{arm}_train{segment}'
            if (folder/'summary.json').exists():
                for rr in C.read(folder/'summary.json').get('rounds',[]):
                    training={r['path']:r for r in rowlist(folder)};rs=[training[x] for x in rr['training_paths']];learning.append(dict(arm=arm,round=rr['round'],fresh_safe=sum(C.M.N.T.Prior.safe(r) for r in rs),fresh_total=len(rs),collision=sum(r['collision'] for r in rs),fit=rr['fit'],model=rr['model']))
    prospective=panel('B0_prospective24');complete &=prospective['verified'];prospective.pop('_rows',None)
    paired={}
    for name in p['panels']:
        a=set(arms['gae95']['panels'][name]['success_ids'] or []);b=set(arms['mc1']['panels'][name]['success_ids'] or []);verified=arms['gae95']['panels'][name]['verified'] and arms['mc1']['panels'][name]['verified'];paired[name]=dict(verified=verified,MC_only=sorted(b-a) if verified else None,GAE_only=sorted(a-b) if verified else None,both=len(a&b) if verified else None)
    result=dict(state=final['state'],complete_all_panels=bool(complete),arms=arms,paired=paired,prospective_B0=prospective,learning_curve=learning,cost=C.read(O/'LEDGER.json'),retained='T698',single_training_seed_pair=True,same_map_exposed_vs_prospective_not_unseen_map=True,no_statistical_training_replication_claim=True,all_originals_preserved=True,epoch=time.time());C.save(D/'PAIRED_RESULTS.json',result)
    lines=['# T721 overnight matched Actor-credit comparison','',f"State {final['state']}; complete final panels {bool(complete)}. One paired training seed; same20198capacity, frozenGRU, commonMC Value targets. Neither method novel; no integration or promotion.",'','| Arm | old403 safe | exposed174 safe | exposure gains/losses | collisions174 | recovery18 | original18 | prospective24 | acquisition | practical |','|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for arm,v in arms.items():
        z=v['panels'];lines.append(f"| {arm} | {z['old403']['safe']} | {z['exposed174']['safe']} | {len(v['gained_exposed'] or [])}/{len(v['lost_exposed'] or [])} | {len(z['exposed174']['collision_ids'] or []) if z['exposed174']['verified'] else 'MISSING'} | {z['recovery18']['safe']} | {v['original18_safe']} | {z['prospective24']['safe']} | {v['acquisition_gate']} | {v['practical_gate']} |")
    lines+=['','Baseline exposed174 safe92/collision41. old403 losses are versus T694 original403; B0 previously lost11. Missing/failed panels are not zero-success evaluations. Exact repeats are audit evidence; all negative results kept. Complete per-ID paired IDs, learning traces and costs are in PAIRED_RESULTS.json. RetainedT698 unchanged. No automatic new method/seed or training beyond this fixed comparison.']
    (D/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    # Simple machine-readable and human-readable learning/cost artifact, no extra science.
    import csv
    with (D/'LEARNING_CURVE.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['arm','round','fresh_safe','fresh_total','collision','value_MSE','approx_KL','clip_fraction'])
        for r in learning:
            dd=r['fit'].get('diagnostic',{});writer.writerow([r['arm'],r['round'],r['fresh_safe'],r['fresh_total'],r['collision'],dd.get('value_MSE'),dd.get('approx_KL'),dd.get('clip_fraction')])
    q=ROOT/'stage/docs/RESEARCH_STATUS.md';q.write_text(f"2026-11-25 T721通宵GAE.95与完整折扣MC对照结束：{final['state']}，完整面板={bool(complete)}；结果见overnight_20261124_25/t721/FINDINGS及PAIRED_RESULTS。单paired训练seed、20198头/冻结GRU，T698保留，无自动新实验。\n\n"+q.read_text())

def backup():
    C.stopped();start=time.time();p=C.root_plan();l=C.read(D/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=900
    files=[O/f for f in ['LEDGER.json','FINAL_REPORT.json','SEALED.json']]+[D/f for f in ['PAIRED_RESULTS.json','FINDINGS.md','LEARNING_CURVE.csv','PREFLIGHT.json','NATIVE_PREFLIGHT.json','STORAGE_BOUND.json']]
    for arm in ['gae95','mc1']:files.append(Path(C.read(O/f'{arm}_train2/summary.json')['model']['path'])) if (O/f'{arm}_train2/summary.json').exists() else None
    for step in p['schedule']:
        files +=[O/step['name']/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    files=[f for f in files if f.exists()];hashes={str(f.relative_to(ROOT)):C.sha(f) for f in files};size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    C.save(D/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),bytes=size,full_histories_local_pending=True,two_final_models_included=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T721 primary180')));signal.alarm(175)
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
    paths=['stage/docs/overnight_20261124_25/t721','stage/experiments/t721_credit_assignment','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']+[str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],check=True,cwd=ROOT);dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,dirty
    subprocess.run(['git','commit','-m','Record T721 complete matched overnight credit comparison'],check=True,cwd=ROOT);target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (D/'final_source.log').open('xb') as f:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize_transport_recovery.py'),'source'],cwd=ROOT,stdout=f,stderr=f)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        receipt=C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target;charge(receipt['remote_commit']);C.save(D/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_VERIFIED',remote_commit=receipt['remote_commit'],root_tree=target))
    else:
        l=C.read(D/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;assert l['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',l);C.save(D/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_PRESERVED',returncode=code))
    C.save(D/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_next_science=True))
if __name__=='__main__':
    try:__import__('source_transport_recovery').main(final=True) if sys.argv[1]=='source' else watch()
    except BaseException:C.save(D/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,epoch=time.time()));raise
