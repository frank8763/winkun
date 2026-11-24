"""Post-exit evidence, finite frontier and one bounded final transport."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t712_finalize_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T712_PRIMARY_0'

def charge(commit):
    l=read(DOC/'TRANSPORT_LEDGER.json')
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(DOC/'TRANSPORT_LEDGER.json',l)

def source():C.sync(final=True)

def analysis():
    p=C.root_plan();paths={n:{s:str(OUT/(s+'_'+n)) for s in ['known','new']} for n in p['arms']}
    paths.update({n:{s:str(ROOT/f) for s,f in folders.items()} for n,folders in p['reused_controls'].items()})
    results={};success={};rows={};complete=True
    for name,folders in paths.items():
        results[name]={}
        for suite,folder in folders.items():
            folder=Path(folder);summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {};audit=read(folder/'audit.json') if (folder/'audit.json').exists() else {}
            valid=summary.get('state')=='COMPLETE' and audit.get('state')=='PASS'
            if not valid:complete=False
            if valid:
                jobs=p['evaluation_jobs'] if suite=='known' else p['new_jobs']
                casefile=folder/'cases.jsonl.gz';cases=[json.loads(x) for x in gzip.decompress(casefile.read_bytes()).decode().splitlines()]
                fresh={r['job_id']:r for r in cases if not r.get('exact')};assert set(fresh)=={j['id'] for j in jobs}
                ids=set(summary['success_ids']);assert ids=={k for k,r in fresh.items() if C.M.N.T.Prior.safe(r)}
                success[name,suite]=ids;rows[name,suite]=fresh
            results[name][suite]=dict(state=summary.get('state','MISSING'),audit=audit.get('state','MISSING'),safe=summary.get('success'),total=summary.get('total'),core407=summary.get('core_success'),old403_lost=summary.get('lost403'),source7_success=summary.get('source_success_ids'),collision_ids=summary.get('collision_ids'),timeout_ids=summary.get('timeout_ids'),boundary_overlap_ids=summary.get('boundary_ids'),model=summary.get('model'),reused=name in p['reused_controls'])
    for suite,jobs in [('known',p['evaluation_jobs']),('new',p['new_jobs'])]:
        available=[name for name in paths if (name,suite) in rows]
        for job in jobs:
            hashes=[tuple(rows[n,suite][job['id']]['array_hashes'][k] for k in ['initial_native_integration','scene_sha256','task_goal_xy']) for n in available]
            assert len(set(hashes))<=1,('Paired reset mismatch',suite,job['id'])
    paired={}
    for reference in ['B0','compact','replay50']:
        for name in paths:
            if name==reference:continue
            for suite in ['known','new']:
                if (name,suite) in success and (reference,suite) in success:
                    a,b=success[name,suite],success[reference,suite];paired[f'{name}_vs_{reference}_{suite}']=dict(gained=sorted(a-b),lost=sorted(b-a),common_safe=len(a&b))
    selections={};order={n:i for i,n in enumerate(p['selection_order'])}
    for limit in p['selection_old_loss_limits']:
        eligible=[n for n in paths if (n,'known') in success and (n,'new') in success and len(results[n]['known']['old403_lost'])<=limit]
        eligible.sort(key=lambda n:(-results[n]['new']['safe'],len(results[n]['known']['old403_lost']),order[n]))
        selections[str(limit)]=dict(selected=eligible[0] if eligible else None,eligible_in_order=eligible,complete_panel=complete,development_selection_not_unbiased_test=True)
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],complete_panel=complete,results=results,paired=paired,selections=selections,
        geometry=read(DOC/'GEOMETRY_CHECK.json'),new_jobs=78,known_jobs=431,new_test_claim=False,retained='T698',epoch=time.time(),cost=read(OUT/'LEDGER.json'))
    save(DOC/'PAIRED_RESULTS.json',report)
    lines=['# T712 finite development retention/transfer comparison','',f"State {report['state']}; complete panel {complete}. T698 retained.",'','| Model | Known audit | Old403 lost | Source7 safe | Development audit | Safe/78 | Collision/78 |','|---|---|---:|---:|---|---:|---:|']
    for n,v in results.items():
        k,d=v['known'],v['new'];lines.append(f"| {n} | {k['audit']} | {len(k['old403_lost']) if k['old403_lost'] is not None else 'MISSING'} | {len(k['source7_success']) if k['source7_success'] is not None else 'MISSING'} | {d['audit']} | {d['safe']} | {len(d['collision_ids']) if d['collision_ids'] is not None else 'MISSING'} |")
    lines+=['','Fixed selection under old403 loss limits4/12/20: '+json.dumps(selections), '',
        'All points and gained/lost IDs, source failures, collision/timeout/boundary overlap and actual costs are retained in PAIRED_RESULTS.json. Missing/failed are not zero success. Development78 is exposed same-map selection data; no fresh-test, independent-map, seed-replication, optimal-BC or novelty claim. Historical B0 is unrefitted; replay50 is the equal-extra-compute control. No automatic next experiment.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T712有限保持/迁移开发对照结束：{report['state']}，完整面板={complete}，结果与固定选择见day_20261124/t712/FINDINGS及PAIRED_RESULTS；78为曝光开发集、非新测试，T698保持，无自动下一科学任务。\n\n"+f.read_text())

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=l['cap']
    p=C.root_plan();files=[OUT/f for f in ['fit/summary.json','fit/audit.json','fit/fit_calls.jsonl','LEDGER.json','SEALED.json','FINAL_REPORT.json']]+[DOC/'PAIRED_RESULTS.json',DOC/'GEOMETRY_CHECK.json']
    for arm in p['arms']:
        files.append(OUT/'fit'/arm/'candidate.pt.gz')
        for suite in ['known','new']:files += [OUT/(suite+'_'+arm)/f for f in ['summary.json','audit.json']]
    missing=[str(f.relative_to(ROOT)) for f in files if not f.exists()];files=[f for f in files if f.exists()]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files};size=sum(f.stat().st_size for f in files)
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,source_bytes=size,files=list(hashes),missing=missing))
    if size>8*2**20:
        save(DOC/'BACKUP_DEFERRED.json',dict(reason='Named primary exceeds fixed8MiB single batch',source_bytes=size,all_local_preserved=True));return
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T712 primary180')));signal.alarm(175)
    try:
        sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
        G.stopped=G.Client.A.stopped=C.stopped;client=G.Client.client();restore=read(G.DOC/'RESTORE_INDEX.json');assert not any(g['group']==GROUP for g in restore['groups'])
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
            result=G.Client.A.archive(GROUP,files,release,min(start+170,C.root_plan()['absolute_all_stop']-330),client)
            C.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json';receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json';members=read(manifest)['files']
            assert {r['path']:r['sha256'] for r in members}==hashes and all(sha(ROOT/r['path'])==r['sha256'] for r in members)
            restore['groups'].append(dict(group=GROUP,receipt=str(receipt),manifest=str(manifest),remote_receipt=result,members=members));restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);save(G.DOC/'RESTORE_INDEX.json',restore)
            save(DOC/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,compressed_bytes=result['bytes'],deleted=0))
    finally:
        signal.alarm(0);l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=time.time()-start;assert l['charged_seconds']+300<=l['cap'];save(DOC/'TRANSPORT_LEDGER.json',l)

def watch():
    p=C.root_plan();save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),one_shot=True,absolute_stop=p['absolute_all_stop']))
    while time.time()<p['absolute_all_stop']-600:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();charge(read(DOC/'TRANSPORT_PENDING.json')['remote_commit']);analysis()
    try:backup()
    except BaseException:save(DOC/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    C.stopped();assert time.time()+300<p['absolute_all_stop']
    paths=['stage/docs/day_20261124','stage/experiments/t712_retention_tradeoff','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    paths += [str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,('Unrelated tracked edits require preserving before final sync',dirty)
    subprocess.run(['git','commit','-m','Record bounded T712 retention tradeoff and all audited outcomes'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (DOC/'final_source.log').open('xb') as log:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=log,stderr=log)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target;charge(receipt['remote_commit'])
        save(DOC/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_VERIFIED',root_tree=target,remote_commit=receipt['remote_commit']))
    else:
        l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;assert l['charged_seconds']<=l['cap'];save(DOC/'TRANSPORT_LEDGER.json',l)
        save(DOC/'FINAL_SOURCE.json',dict(state='FAILED_ALL_LOCAL_PRESERVED',returncode=code,no_retry=True))
    save(DOC/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_automatic_next_science=True))

if __name__=='__main__':
    try:source() if sys.argv[1]=='source' else watch()
    except BaseException:save(DOC/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),epoch=time.time(),all_local_preserved=True));raise
