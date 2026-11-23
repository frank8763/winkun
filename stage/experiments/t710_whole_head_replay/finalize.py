"""One-shot post-exit analysis, named evidence readback and final source."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,signal,subprocess,sys,time,traceback,collections
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t710_final_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T710_PRIMARY_0'

def charge(commit):
    p=DOC/'TRANSPORT_LEDGER.json';l=read(p)
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(p,l)

def source():C.sync(final=True)

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+180+300<=l['cap']
    choices=[OUT/f for f in ['fit/nominal.pt.gz','fit/induced.pt.gz','fit/summary.json','fit/audit.json','fit/fit_calls.jsonl','fit_main.log','fit_audit.log','LEDGER.json','SEALED.json','FINAL_REPORT.json','status.json']]+[DOC/'PAIRED_RESULTS.json']
    for name in ['known_nominal','new_nominal','known_induced','new_induced']:choices += [OUT/name/f for f in ['summary.json','audit.json']]
    files=[p for p in choices if p.exists()];missing=[str(p.relative_to(ROOT)) for p in choices if not p.exists()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};size=sum(p.stat().st_size for p in files)
    assert size<=8*2**20,'One fixed primary set exceeds8MiB; retain local, no split or rescue'
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),source_bytes=size,missing=missing))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T710 primary180')));signal.alarm(175)
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
    for arm in ['nominal','induced']:
        results[arm]={}
        for suite in ['known','new']:
            folder=OUT/(suite+'_'+arm);s=read(folder/'summary.json') if (folder/'summary.json').exists() else {};a=read(folder/'audit.json') if (folder/'audit.json').exists() else {}
            results[arm][suite]=dict(state=s.get('state','MISSING'),audit=a.get('state','MISSING'),safe=s.get('success'),total=s.get('total'),core_success=s.get('core_success'),
                lost403=s.get('lost403'),source_success_ids=s.get('source_success_ids'),collision=len(s.get('collision_ids',[])),timeout=len(s.get('timeout_ids',[])),boundary=len(s.get('boundary_ids',[])))
            if s.get('state')=='COMPLETE' and a.get('state')=='PASS':success[(arm,suite)]=set(s['success_ids'])
            f=folder/'cases.jsonl.gz';rows[(arm,suite)]={r['job_id']:r for r in [json.loads(x) for x in gzip.decompress(f.read_bytes()).decode().splitlines()] if not r.get('exact')} if f.exists() else {}
    original=read(ROOT/'stage/runs/t704_frozen_new_starts/new_refined108/summary.json');reference=set(original['success_ids']);paired={}
    for arm in ['nominal','induced']:
        if (arm,'new') in success:
            value=success[(arm,'new')];paired[arm+'_vs_original_refined108']=dict(gained_ids=sorted(value-reference),lost_ids=sorted(reference-value),common_safe=len(value&reference))
    for suite in ['known','new']:
        if all((a,suite) in success for a in ['nominal','induced']):
            x,y=success[('induced',suite)],success[('nominal',suite)];paired['induced_vs_nominal_'+suite]=dict(gained_ids=sorted(x-y),lost_ids=sorted(y-x),common_safe=len(x&y))
    original_oldlost=set(read(ROOT/'stage/runs/t701r_native_entry/known_replay_refined108/summary.json')['lost403'])
    for arm in ['nominal','induced']:
        if (arm,'known') in success:
            current=set(p['required403_ids'])-success[(arm,'known')]
            paired[arm+'_old403_vs_original']=dict(regained_ids=sorted(original_oldlost-current),newly_lost_ids=sorted(current-original_oldlost),still_lost_ids=sorted(current&original_oldlost))
    initial_exact=None
    if all((a,'new') in success for a in ['nominal','induced']):
        oldrows={r['job_id']:r for r in [json.loads(x) for x in gzip.decompress((ROOT/'stage/runs/t704_frozen_new_starts/new_refined108/cases.jsonl.gz').read_bytes()).decode().splitlines()] if not r.get('exact')}
        for j in p['new_jobs']:
            for key in ['initial_native_integration','scene_sha256','task_goal_xy']:
                assert rows[('nominal','new')][j['id']]['array_hashes'][key]==rows[('induced','new')][j['id']]['array_hashes'][key]==oldrows[j['id']]['array_hashes'][key]
        initial_exact=True
    groups={}
    for field in ['neighborhood_parent_id','prospective_ring','evaluation_cluster']:
        table={}
        for job in p['new_jobs']:
            entry=table.setdefault(str(job[field]),dict(planned=0,models={}));entry['planned']+=1
            for arm in ['nominal','induced']:
                r=rows[(arm,'new')].get(job['id']);v=entry['models'].setdefault(arm,dict(executed=0,safe=0,collision=0,timeout=0,boundary=0))
                if r is not None:
                    v['executed']+=1;v['safe']+=int(C.M.N.T.Prior.safe(r));v['collision']+=int(bool(r['collision']));v['timeout']+=int(not r['success'] and not r['collision']);v['boundary']+=int(bool(r.get('field_boundary')) and 'fullfield_start_pose' in job)
        groups[field]=table
    prior_coverage=read(ROOT/'stage/docs/day_20261123/t709/PAIRED_RESULTS.json')
    fixed_readout_refs=dict(nominal=dict(lost403=15,joint224=97,source7=6),induced=prior_coverage.get('results',{}).get('induced',{}))
    report=dict(fixed_readout_references=fixed_readout_refs,state=read(OUT/'FINAL_REPORT.json')['state'],results=results,paired=paired,groups=groups,initial_hashes_equal_T704=initial_exact,retained='T698',epoch=time.time(),
        original_refined108=dict(new_safe=97,total=224,lost403=11,added_tensor_bytes=161584),exposed_regression_not_new_holdout=True,new_Adam=0,new_RL=0)
    save(DOC/'PAIRED_RESULTS.json',report)
    lines=['# T710 Whole-head nominal versus induced-state replay','',f"State {report['state']}; T698 retained. Originalrefined108 historical old403lost11,new224safe97; T705constrained12/100. Both arms same161584 addedtensorbytes.",'',
      '| Arm | Known audit | Core/407 | Old403 lost | Source7 safe | Joint224 audit | Joint224 safe |','|---|---|---:|---:|---:|---|---:|']
    for arm,v in results.items():
        k,n=v['known'],v['new'];lines.append(f"| {arm} | {k['audit']} | {k['core_success']} | {len(k['lost403']) if k['lost403'] is not None else 'MISSING'} | {len(k['source_success_ids']) if k['source_success_ids'] is not None else 'MISSING'} | {n['audit']} | {n['safe']} |")
    lines+=['','Full paired IDs/grouped parent/ring/sector/shared5m tables: PAIRED_RESULTS.json. Full-head nonlinear LBFGS has a finite matched160-gradient cap/arm, not global convergence. Both complete suites evaluated without a forgetting gate.',
      'Only coverage differs within the two whole-head arms: same B0 initial full20198-parameter residual head/160-gradient budget/ridge/data weighting, frozen T694 parent. Deployment161584 addedbytes unchanged. Historical T705/T709 fixed-readout cells also differ in optimizer and trainable parameter count; descriptive references, not a pure representation interaction. T691/T668 teacher-recovery failures remain. No full DAgger/DER replication, RL,novelty,unseenmap or blindtest claim. Related starts are not independent worlds.',
      'Independent construction/state/payload/nativearray audits do not reintegrate all trajectories. Model/primary remote coverage only if PRIMARY_BACKUP verifies, allraw local retained0deleted; no automatic nextjob.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-23 T710同容量全小网络头名义/诱导回放比较结束：{report['state']}，完整成对结果/原403保持/224曝光回归见overnight_20261123_24/t710/FINDINGS。T698保持，无自动下一任务。\n\n"+f.read_text())

def watch():
    p=read(OUT/'protocol.json');save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),one_shot=True,epoch=time.time(),absolute_stop=p['absolute_all_stop']))
    while time.time()<p['absolute_all_stop']-10:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();analysis();pending=read(DOC/'TRANSPORT_PENDING.json');charge(pending['remote_commit']);pending.update(accounted=True);save(DOC/'TRANSPORT_PENDING.json',pending)
    if read(DOC/'TRANSPORT_LEDGER.json')['charged_seconds']+180+300<=read(DOC/'TRANSPORT_LEDGER.json')['cap']:
        try:backup()
        except BaseException:save(DOC/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    else:save(DOC/'BACKUP_DEFERRED.json',dict(state='PRIMARY_DEFERRED_TO_T711_WITHIN_FIXED_TRANSPORT_CAP',reason='Original failed source300 plus successful pre300 plus reserved final300 exhaust T710900 cap; all local bytes preserved',no_new_native_or_learning=True))
    C.stopped()
    paths=['stage/docs/overnight_20261123_24/t710','stage/experiments/t710_whole_head_replay','stage/docs/RESEARCH_STATUS.md',
        'stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    for name in [f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json',f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json']:
        if (ROOT/name).exists():paths.append(name)
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record T710 whole compact head matched replay and paired outcomes'],cwd=ROOT,check=True)
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
