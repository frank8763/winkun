"""One-shot post-exit analysis, named evidence readback and final source."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,signal,subprocess,sys,time,traceback,collections
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t709_final_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T709_PRIMARY_0'

def charge(commit):
    p=DOC/'TRANSPORT_LEDGER.json';l=read(p)
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(p,l)

def source():C.sync(final=True)

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+180+300<=l['cap']
    choices=[OUT/f for f in ['fit/induced.pt.gz','fit/summary.json','fit/audit.json','fit/fit_calls.jsonl','fit_main.log','fit_audit.log','LEDGER.json','SEALED.json','FINAL_REPORT.json','status.json']]+[DOC/'PAIRED_RESULTS.json']
    for name in ['known_induced','new_induced']:choices += [OUT/name/f for f in ['summary.json','audit.json']]
    files=[p for p in choices if p.exists()];missing=[str(p.relative_to(ROOT)) for p in choices if not p.exists()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};size=sum(p.stat().st_size for p in files)
    assert size<=8*2**20,'One fixed primary set exceeds8MiB; retain local, no split or rescue'
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),source_bytes=size,missing=missing))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T709 primary180')));signal.alarm(175)
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
    p=read(OUT/'protocol.json');results={};success={};rows={}
    roots=dict(induced=OUT,control=ROOT/'stage/runs/t705_constrained_replay',original_known=ROOT/'stage/runs/t701r_native_entry',original_new=ROOT/'stage/runs/t704_frozen_new_starts')
    paths=dict(induced=dict(known=OUT/'known_induced',new=OUT/'new_induced'),
        control=dict(known=roots['control']/'known_balanced',new=roots['control']/'new_balanced'),
        original=dict(known=roots['original_known']/'known_replay_refined108',new=roots['original_new']/'new_refined108'))
    for model,folders in paths.items():
        results[model]={}
        for suite,folder in folders.items():
            s=read(folder/'summary.json') if (folder/'summary.json').exists() else {};a=read(folder/'audit.json') if (folder/'audit.json').exists() else {}
            results[model][suite]=dict(state=s.get('state','MISSING'),audit=a.get('state','MISSING'),safe=s.get('success'),total=s.get('total'),core_success=s.get('core_success'),
                lost403=s.get('lost403'),source_success_ids=s.get('source_success_ids'),collision=len(s.get('collision_ids',[])),timeout=len(s.get('timeout_ids',[])),boundary=len(s.get('boundary_ids',[])))
            if s.get('state')=='COMPLETE' and a.get('state')=='PASS':success[(model,suite)]=set(s['success_ids'])
            f=folder/'cases.jsonl.gz';rows[(model,suite)]={r['job_id']:r for r in [json.loads(x) for x in gzip.decompress(f.read_bytes()).decode().splitlines()] if not r.get('exact')} if f.exists() else {}
    paired={}
    for reference in ['control','original']:
        for suite in ['known','new']:
            if all((name,suite) in success for name in ['induced',reference]):
                x,y=success[('induced',suite)],success[(reference,suite)];paired['induced_vs_'+reference+'_'+suite]=dict(gained_ids=sorted(x-y),lost_ids=sorted(y-x),common_safe=len(x&y))
        if all((name,'known') in success for name in ['induced',reference]):
            current=set(p['required403_ids'])-success[('induced','known')];oldlost=set(p['required403_ids'])-success[(reference,'known')]
            paired['induced_old403_vs_'+reference]=dict(regained_ids=sorted(oldlost-current),newly_lost_ids=sorted(current-oldlost),still_lost_ids=sorted(current&oldlost))
    initial_exact=None
    if ('induced','new') in success:
        for j in p['new_jobs']:
            for key in ['initial_native_integration','scene_sha256','task_goal_xy']:
                assert rows[('induced','new')][j['id']]['array_hashes'][key]==rows[('control','new')][j['id']]['array_hashes'][key]==rows[('original','new')][j['id']]['array_hashes'][key]
        initial_exact=True
    groups={}
    for field in ['neighborhood_parent_id','prospective_ring','evaluation_cluster']:
        table={}
        for job in p['new_jobs']:
            entry=table.setdefault(str(job[field]),dict(planned=0,models={}));entry['planned']+=1
            for model in paths:
                r=rows[(model,'new')].get(job['id']);v=entry['models'].setdefault(model,dict(executed=0,safe=0,collision=0,timeout=0,boundary=0))
                if r is not None:
                    v['executed']+=1;v['safe']+=int(C.M.N.T.Prior.safe(r));v['collision']+=int(bool(r['collision']));v['timeout']+=int(not r['success'] and not r['collision']);v['boundary']+=int(bool(r.get('field_boundary')) and 'fullfield_start_pose' in job)
        groups[field]=table
    covered=set(r['job_id'] for r in p['induced_selection']['rows']);old=set(p['required403_ids']);coverage={}
    for model in paths:
        if (model,'known') in success:
            coverage[model]={label:dict(total=len(panel),safe=len(panel & success[(model,'known')]),lost=sorted(panel-success[(model,'known')])) for label,panel in [('covered51',covered),('uncovered352',old-covered)]}
    pareto={}
    if all(('induced',s) in success for s in ['known','new']):
        for ref in ['control','original']:
            a,b=results['induced'],results[ref];la,lb=len(a['known']['lost403']),len(b['known']['lost403']);na,nb=a['new']['safe'],b['new']['safe']
            pareto[ref]=dict(old_loss_delta=la-lb,joint_success_delta=na-nb,dominates=la<=lb and na>=nb and (la<lb or na>nb),dominated=la>=lb and na<=nb and (la>lb or na<nb))
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],results=results,paired=paired,groups=groups,coverage=coverage,pareto=pareto,initial_hashes_equal_historical=initial_exact,retained='T698',epoch=time.time(),
        matched_deployed_added_tensor_bytes=161584,selected_old_IDs=51,histories=102,old_ID_weights_unchanged=True,offline_one_fixed_snapshot_not_iterative_DAgger=True,teacher_recovery_not_demonstrated=True,exposed_regression_not_new_holdout=True,new_Adam=0,new_RL=0)
    save(DOC/'PAIRED_RESULTS.json',report)
    lines=['# T709 fixed-capacity learner-induced replay coverage','',f"State {report['state']}; T698 retained. Same161584 addedtensorbytes; same final109x2 readout, hidden/body frozen. Historical T705 balanced control reconstructed bitwise before native reuse.",'',
      '| Model | Known audit | Core/407 | Old403 lost | Source7 safe | Joint224 audit | Joint224 safe |','|---|---|---:|---:|---:|---|---:|']
    for model,v in results.items():
        k,n=v['known'],v['new'];lines.append(f"| {model} | {k['audit']} | {k['core_success']} | {len(k['lost403']) if k['lost403'] is not None else 'MISSING'} | {len(k['source_success_ids']) if k['source_success_ids'] is not None else 'MISSING'} | {n['audit']} | {n['safe']} |")
    lines+=['','Full paired success/regained/newloss IDs and parent/ring/sector/shared5m grouping: PAIRED_RESULTS.json. No weighted scalar or zero-forgetting gate. Both suites already exposed regression, not independent maps or fresh holdout.',
      'One data-coverage ablation using all102 pre-existing full old-ID histories (51/403 IDs x two T708 policies), no success/failure filtering. Per old ID original .5 +mean-induced .25 +worst-induced .25 for covered IDs, original1 elsewhere. Source7 weight/ridge/architecture unchanged. All supervised induced labels are frozen T694 ownhistory rawactions; not evidence of teacher recovery in those states.',
      'T691 induced-supervision/KL failure remains counterevidence; this fixed-feature exact-convex-readout setting removes that optimizer/KLgate, not a new method or full DAgger reproduction. Conditional coverage bias and unseen current-candidate occupancy remain. Independent saved-payload audits do not reintegrate every physics trajectory. All original raw retained0deleted; no automatic nextscience.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-23 T709同容量诱导状态回放覆盖对照结束：{report['state']}，完整成对保持-迁移结果见day_20261123/t709/FINDINGS。T698保持，无自动下一任务。\n\n"+f.read_text())

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
    paths=['stage/docs/day_20261123/t709','stage/experiments/t709_induced_replay','stage/docs/RESEARCH_STATUS.md',
        'stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    for name in [f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json',f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json']:
        if (ROOT/name).exists():paths.append(name)
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record T709 fixed-capacity induced-state coverage and paired outcomes'],cwd=ROOT,check=True)
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
