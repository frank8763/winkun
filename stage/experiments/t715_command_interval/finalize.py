"""Post-exit evidence, finite frontier and one bounded final transport."""
from pathlib import Path
import fcntl,gzip,importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t715_finalize_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T715_PRIMARY_0'

def charge(commit):
    l=read(DOC/'TRANSPORT_LEDGER.json')
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(DOC/'TRANSPORT_LEDGER.json',l)

def source():C.sync(final=True)

def analysis():
    p=C.root_plan(); paths={n:{s:str(OUT/(s+'_'+n)) for s in ['known','new','acquisition','exposed','test']} for n in p['arms']}
    for n,folders in p['reused_controls'].items():paths[n]={**{s:str(ROOT/f) for s,f in folders.items()},'test':str(OUT/('test_'+n))}
    jobs={'known':p['evaluation_jobs'],'new':p['new_jobs'],'acquisition':p['acquisition_jobs'],'test':p['test_jobs'],'exposed':p['exposed_jobs']}
    results={}; success={}; rows={}; complete=True
    for name,folders in paths.items():
        results[name]={}
        for suite,folder in folders.items():
            folder=Path(folder);sm=read(folder/'summary.json') if (folder/'summary.json').exists() else {};au=read(folder/'audit.json') if (folder/'audit.json').exists() else {}
            valid=sm.get('state')=='COMPLETE' and au.get('state')=='PASS'
            if not valid:complete=False
            if valid:
                cases=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
                fresh={r['job_id']:r for r in cases if not r.get('exact')};assert set(fresh)=={j['id'] for j in jobs[suite]}
                ids=set(sm['success_ids']);assert ids=={k for k,r in fresh.items() if C.M.N.T.Prior.safe(r)}
                rows[name,suite]=fresh;success[name,suite]=ids
            results[name][suite]=dict(state=sm.get('state','MISSING'),audit=au.get('state','MISSING'),safe=sm.get('success'),total=sm.get('total'),core407=sm.get('core_success'),old403_lost=sm.get('lost403'),source7_success=sm.get('source_success_ids'),collision_ids=sm.get('collision_ids'),timeout_ids=sm.get('timeout_ids'),boundary_overlap_ids=sm.get('boundary_ids'),model=sm.get('model'),reused=name in p['reused_controls'] and suite!='test')
    base={r['job_id']:r['baseline'] for r in p['source_selection']['rows']}
    assert all(not C.M.N.T.Prior.safe(r) for r in base.values())
    rows['B0','acquisition']=base;success['B0','acquisition']=set()
    results['B0']['acquisition']=dict(state='REUSED_AUDITED_SELECTION_BASELINES',audit='PASS',safe=0,total=18,reused=True)
    
    for suite,js in jobs.items():
        available=[n for n in paths if (n,suite) in rows]
        for job in js:
            hashes=[tuple(rows[n,suite][job['id']]['array_hashes'][k] for k in ['initial_native_integration','scene_sha256','task_goal_xy']) for n in available]
            assert len(set(hashes))<=1,('Paired reset mismatch',suite,job['id'])
    paired={}
    for reference in ['B0','raw']:
        for name in paths:
            if name==reference:continue
            for suite in jobs:
                if (name,suite) in success and (reference,suite) in success:
                    a,b=success[name,suite],success[reference,suite];paired[f'{name}_vs_{reference}_{suite}']=dict(gained=sorted(a-b),lost=sorted(b-a),common_safe=len(a&b))
    bylayout={}
    for n in paths:
        if (n,'test') in rows:
            bylayout[n]={}
            for layout in p['layout_rules']:
                selected=[j['id'] for j in p['test_jobs'] if j['layout_id']==layout]
                bylayout[n][layout]=dict(total=len(selected),safe=sum(j in success[n,'test'] for j in selected),collision=sum(bool(rows[n,'test'][j]['collision']) for j in selected))
    gate=None
    if complete:
        sk=results['interval']['known'];st=results['interval']['test']
        gate=dict(old_loss_within20=len(sk['old403_lost'])<=20,
          safe_above_B0_and_raw=all(st['safe']>results[n]['test']['safe'] for n in ['B0','raw']),
          collisions_no_more_than_B0_and_raw=all(len(st['collision_ids'])<=len(results[n]['test']['collision_ids']) for n in ['B0','raw']),
          acquisition_above_raw=results['interval']['acquisition']['safe']>results['raw']['acquisition']['safe'])
        gate['all']=all(gate.values())
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],complete_scheduled_panel=complete,results=results,paired=paired,per_layout=bylayout,
        practical_finite_gate=gate,predeclared_gate=p['practical_finite_gate'],retained='T698',epoch=time.time(),cost=read(OUT/'LEDGER.json'),geometry=read(DOC/'GEOMETRY_QUALIFICATION.json'),fresh_layouts=3,fresh_starts=72,selected_sources=18,source_scope='exposed_development',no_novel_algorithm_or_independent_world_claim=True)
    save(DOC/'PAIRED_RESULTS.json',report)
    lines=['# T715 fixed-capacity executed-command interval supervision','',f"State {report['state']}; complete scheduled panel {complete}. T698 retained.",'','| Model | Old403 lost | Original7 safe | Exposed safe/78 | Acquired safe/18 | Exposed layout safe/72 | Fresh safe/72 | Fresh collisions/72 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,v in results.items():
        k,d,a,e,t=[v[s] for s in ['known','new','acquisition','exposed','test']]
        lines.append(f"| {n} | {len(k['old403_lost']) if k['old403_lost'] is not None else 'MISSING'} | {len(k['source7_success']) if k['source7_success'] is not None else 'MISSING'} | {d['safe']} | {a['safe'] if a['safe'] is not None else 'NOT SCHEDULED'} | {e['safe']} | {t['safe']} | {len(t['collision_ids']) if t['collision_ids'] is not None else 'MISSING'} |")
    lines+=['','Predeclared finite practical gate: '+json.dumps(gate), '',
      'All arms, audits, paired gained/lost IDs, per-layout outcomes, collision/timeout/boundary overlap and costs are in PAIRED_RESULTS.json. The fixed head has20198 trainable parameters and161584 added tensor bytes; parent2621976 bytes is shared. Raw is the exact audited T714 success candidate, same initialization/data/weights/closure cap; treatment changes all-category raw targets to command-canonical intervals including small old unsaturated rounding differences. Source acquisition is training evidence, exposed78 is development; three related perturbed layouts are a limited prospective test with unknown path feasibility. Occupancy/data are fixed across raw and interval; supervision semantics change. This is a known clipping-aware supervision adaptation, not a new algorithm. No new RL, algorithmic novelty, statistical population claim, automatic promotion or automatic next experiment. Missing is not zero.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T715同容量执行命令区间监督对照结束：{report['state']}，完整预定面板={complete}；已曝光18来源验收和新三布局72起点成对结果见day_20261124/t715/FINDINGS及PAIRED_RESULTS。T698保持，无自动下一任务。\n\n"+f.read_text())

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=l['cap']
    p=C.root_plan();files=[OUT/f for f in ['fit/summary.json','fit/audit.json','fit/fit_calls.jsonl','LEDGER.json','SEALED.json','FINAL_REPORT.json']]+[DOC/'PAIRED_RESULTS.json',DOC/'GEOMETRY_QUALIFICATION.json']
    for arm in p['arms']:
        files.append(OUT/'fit'/arm/'candidate.pt.gz')
        for suite in ['known','new','acquisition','exposed']:files += [OUT/(suite+'_'+arm)/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    for arm in p['transfer_models']:files += [OUT/('test_'+arm)/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    missing=[str(f.relative_to(ROOT)) for f in files if not f.exists()];files=[f for f in files if f.exists()]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files};size=sum(f.stat().st_size for f in files)
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,source_bytes=size,files=list(hashes),missing=missing))
    if size>8*2**20:
        save(DOC/'BACKUP_DEFERRED.json',dict(reason='Named primary exceeds fixed8MiB single batch',source_bytes=size,all_local_preserved=True));return
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T715 primary180')));signal.alarm(175)
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
    paths=['stage/docs/day_20261124/t715','stage/experiments/t715_command_interval','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    paths += [str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,('Unrelated tracked edits require preserving before final sync',dirty)
    subprocess.run(['git','commit','-m','Record bounded T715 command interval supervision and all audited outcomes'],cwd=ROOT,check=True)
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
