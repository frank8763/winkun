"""One-shot post-exit analysis, named evidence readback and final source."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,signal,subprocess,sys,time,traceback,collections
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t713_final_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T713_PRIMARY_0'

def charge(commit):
    p=DOC/'TRANSPORT_LEDGER.json';l=read(p)
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(p,l)

def source():C.sync(final=True)

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+180+300<=l['cap']
    choices=[OUT/f for f in ['fit/summary.json','fit/audit.json','LEDGER.json','SEALED.json','FINAL_REPORT.json','status.json']]+[DOC/'INITIALS.json',DOC/'PAIRED_RESULTS.json']
    for name in ['new_'+n for n in read(OUT/'fit/summary.json')['finals']]:choices += [OUT/name/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    choices += [DOC/'LAYOUTS.json',DOC/'GEOMETRY_QUALIFICATION.json']
    files=[p for p in choices if p.exists()];missing=[str(p.relative_to(ROOT)) for p in choices if not p.exists()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};size=sum(p.stat().st_size for p in files)
    assert size<=8*2**20,'One fixed primary set exceeds8MiB; retain local, no split or rescue'
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),source_bytes=size,missing=missing))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T713 primary180')));signal.alarm(175)
    try:
        sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
        G.stopped=G.Client.A.stopped=C.stopped;client=G.Client.client();restore=read(G.DOC/'RESTORE_INDEX.json')
        assert not any(g['group']==GROUP for g in restore['groups'])
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
            result=G.Client.A.archive(GROUP,files,release,min(start+170,read(OUT/'protocol.json')['absolute_all_stop']-330),client)
            C.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json';receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json'
            members=read(manifest)['files'];assert {r['path']:r['sha256'] for r in members}==hashes
            assert all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==r['bytes'] for r in members)
            restore['groups'].append(dict(group=GROUP,receipt=str(receipt),manifest=str(manifest),remote_receipt=result,members=members))
            restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);save(G.DOC/'RESTORE_INDEX.json',restore)
            save(DOC/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,compressed_bytes=result['bytes'],missing=missing,deleted=0))
    finally:
        signal.alarm(0);l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=time.time()-start
        assert l['charged_seconds']+300<=l['cap'];save(DOC/'TRANSPORT_LEDGER.json',l)

def analysis():
    p=read(OUT/'protocol.json');initials=read(DOC/'INITIALS.json');jobs={j['id']:j for j in p['new_jobs']}
    names=['parent','compact','B0'];models={};sets={};by_model={}
    for name in names:
        folder=OUT/('new_'+name);sf,af=folder/'summary.json',folder/'audit.json'
        s=read(sf) if sf.exists() else {};a=read(af) if af.exists() else {}
        row=dict(model=name,state=s.get('state','MISSING'),audit=a.get('state','MISSING'),success=s.get('success'),total=s.get('total'),
            collision=len(s.get('collision_ids',[])),timeout=len(s.get('timeout_ids',[])),boundary=len(s.get('boundary_ids',[])),
            physics=s.get('physics'),commands=s.get('commands'),model_checkpoint=s.get('model'))
        models[name]=row
        if s.get('state')=='COMPLETE' and a.get('state')=='PASS':sets[name]=set(s['success_ids'])
        cf=folder/'cases.jsonl.gz'
        rows=[json.loads(x) for x in gzip.decompress(cf.read_bytes()).decode().splitlines()] if cf.exists() else []
        by_model[name]={r['job_id']:r for r in rows if not r.get('exact')}
    paired={}
    comparisons=[('B0','parent'),('compact','parent'),('B0','compact')]
    for candidate_name,reference_name in comparisons:
        if candidate_name in sets and reference_name in sets:
            candidate,reference=sets[candidate_name],sets[reference_name]
            paired[candidate_name+'_vs_'+reference_name]=dict(gained_ids=sorted(candidate-reference),lost_ids=sorted(reference-candidate),common_success=len(candidate&reference),common_failure=len(set(jobs)-(candidate|reference)))
    groups={}
    for group_key in ['neighborhood_parent_id','prospective_ring','evaluation_cluster']:
        table={}
        for jid,job in jobs.items():
            group=str(job[group_key]);entry=table.setdefault(group,dict(planned=0,models={}))
            entry['planned']+=1
            for name in names:
                r=by_model[name].get(jid)
                value=entry['models'].setdefault(name,dict(executed=0,safe=0,collision=0,timeout=0,boundary=0,administrative_censor=0))
                if r is None:continue
                value['executed']+=1;value['safe']+=int(C.M.N.T.Prior.safe(r));value['collision']+=int(bool(r['collision']))
                value['timeout']+=int(not r['success'] and not r['collision']);value['boundary']+=int(bool(r.get('field_boundary')) and 'fullfield_start_pose' in job)
                value['administrative_censor']+=int(bool(r.get('administrative_censor')))
        groups[group_key]=table
    if sets:
        # All available complete models must share the policy-independent reset hash.
        for jid in jobs:
            keys=['initial_native_integration','scene_sha256','task_goal_xy']
            hashes=[tuple(by_model[n][jid]['array_hashes'][k] for k in keys) for n in sets]
            assert len(set(hashes))==1,(jid,'paired initial mismatch')
    historical={}
    known_paths={'compact':ROOT/'stage/runs/t703_projected_local/known_proj8','B0':ROOT/'stage/runs/t701r_native_entry/known_replay_refined108','constrained':ROOT/'stage/runs/t705_constrained_replay/known_constrained','coverage':ROOT/'stage/runs/t709_induced_replay/known_induced','whole_nominal':ROOT/'stage/runs/t710_whole_head_replay/known_nominal','whole_induced':ROOT/'stage/runs/t710_whole_head_replay/known_induced'}
    for name,folder in known_paths.items():
        sf,af=folder/'summary.json',folder/'audit.json';s=read(sf) if sf.exists() else {};a=read(af) if af.exists() else {}
        historical[name]=dict(state=s.get('state','MISSING'),audit=a.get('state','MISSING'),core407=s.get('core_success'),old403_losses=len(s['lost403']) if s.get('lost403') is not None else None,source7_success=len(s['source_success_ids']) if s.get('source_success_ids') is not None else None,summary=str(sf),model_sha256=s.get('model',{}).get('sha256'))
        if s.get('state')=='COMPLETE' and a.get('state')=='PASS' and models[name].get('model_checkpoint'):
            assert s['model']['sha256']==models[name]['model_checkpoint']['sha256']
    historical={k:v for k,v in historical.items() if k in names}
    historical['parent']=dict(old403_losses=0,core407=390,supp24=13,source7_success=0,scope='T694 retained core390 plus audited T69513supp; seven T696 source IDs were outside this403, not a newly run known suite')
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],models=models,historical_known=historical,paired=paired,groups=groups,
        planned_initials=72,valid_initials=len(jobs),invalid=initials['invalid'],duplicates=initials['duplicates'],
        paired_initial_hashes_exact=bool(sets),new_learning=0,retained='T698',
        interpretation='Three fixed source-derived obstacle translations; held-out geometry, path feasibility unknown, not independent worlds or training replication',epoch=time.time())
    save(DOC/'PAIRED_RESULTS.json',report);save(DOC/'FINDINGS.json',report)
    lines=['# T713 frozen prospective initial-state comparison','',f"State: {report['state']}. New learning0. Retained T698 unchanged.",'',
        f"Planned72 initial poses/model; valid{len(jobs)}, invalid{len(initials['invalid'])}, duplicate{len(initials['duplicates'])}; no replacements.",'',
        '| Model | Audit | Safe | Collision | Timeout | Boundary(overlap) |','|---|---|---:|---:|---:|---:|']
    for name,r in models.items():lines.append(f"| {name} | {r['audit']} | {r['success']}/{r['total']} | {r['collision']} | {r['timeout']} | {r['boundary']} |")
    lines+=['','Historical old403 losses / seven-source success (complete prior audits only):']
    for name,value in historical.items():lines.append(f"- {name}: oldlost{value.get('old403_losses')},source7{value.get('source7_success')},knownaudit{value.get('audit','historical composite')}.")
    lines+=['','Predeclared paired gains/losses:']
    for name,r in paired.items():lines.append(f"- vs{name}: gains{len(r['gained_ids'])}, losses{len(r['lost_ids'])}, common safe{r['common_success']}.")
    lines+=['','Complete per-parent/ring/sector/shared-source-snapshot tables and actual IDs are in PAIRED_RESULTS.json. Related starts are not independent environments; no binomial or pseudo-independent bootstrap significance claim.',
        'Fixed primary comparison B0 versus compact, both versus parent. Three deterministic source-derived circle layouts; no task replacement or tuning. Endpoint validity is not path feasibility. Per-layout paired results are descriptive; no independent benchmark world, population significance or algorithm novelty claim. No automatic next job.',
        'Report historical known431/old403/source7 outcomes alongside fresh72; all prior audits reused by exactmodelSHA, not retested here. Addedbytes parent0,compact157684, allMLP161584 plus2621976 common parent. No independent maps or trainingseed replication.',
        'Each valid fresh trajectory saved full; chosen exact repeats compact. Independent saved-payload/owncarry/integral/initial/exact audit, not reintegration of every trajectory. Missing/failed checks remain missing/failed.',
        'All local raw retained. Named primary evidence remote coverage only if PRIMARY_BACKUP passes; remaining full raw localpending.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-23 T713固定七模型新起点评测结束：{report['state']}，独立audit与完整成对/分组结果见overnight_20261123_24/t713/FINDINGS及PAIRED_RESULTS。0学习，T698保持；三固定未曝光障碍布局，路径可行性未知，非独立世界总体样本，不自动下一任务。\n\n"+f.read_text())

def watch():
    p=read(OUT/'protocol.json');save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),one_shot=True,epoch=time.time(),absolute_stop=p['absolute_all_stop']))
    while time.time()<p['absolute_all_stop']-10:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();analysis();pending=read(DOC/'TRANSPORT_PENDING.json');charge(pending['remote_commit']);pending.update(accounted=True);save(DOC/'TRANSPORT_PENDING.json',pending)
    try:backup()
    except BaseException:save(DOC/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    C.stopped()
    paths=['stage/docs/day_20261124/t713','stage/experiments/t713_unseen_layouts','stage/docs/RESEARCH_STATUS.md',
        'stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    for name in [f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json',f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json']:
        if (ROOT/name).exists():paths.append(name)
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record T713 fixed three-model held-out layouts'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (DOC/'final_source.log').open('xb') as log:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=log,stderr=log)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==target;charge(r['remote_commit'])
        save(DOC/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_SOURCE300_ACCOUNTED',remote_commit=r['remote_commit'],root_tree=target))
    else:
        l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;save(DOC/'TRANSPORT_LEDGER.json',l)
        save(DOC/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_PRESERVED',returncode=code,no_retry=True))
    save(DOC/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_next_science_or_recurring_automation=True))

if __name__=='__main__':
    try:source() if sys.argv[1]=='source' else watch()
    except BaseException:save(DOC/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),epoch=time.time(),all_local_preserved=True));raise
