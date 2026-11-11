"""Next direction: one bounded old-task V-trace rehearsal pilot, not A/B."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,gzip,io,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t486_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT,DOC=T.OUT,ROOT/'stage/docs/review_20261111'


def prepare():
    import t486_cache as K
    import t486_storage as S
    assert not OUT.exists();K._idle()
    previous=ROOT/'stage/runs/t485_afternoon_recovery'
    report=read(DOC/'T485_FINAL_REPORT.json')
    assert report['state']=='COMPLETE_AUDITED_LOCAL' and report['remote']==K.COMPLETE
    assert read(previous/'pair/audit.json')['state']=='PASS'
    assert report['summaries']['pair']['result']['chosen_arm'] is None,'Retained progress: stop failure relay'
    base=report['selected'];assert [base[k]['success'] for k in ['wide8','wide6','old275']]==[8,53,274]
    stop=datetime(2026,11,11,20,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    assert time.time()+6600<stop,'Insufficient complete capture+one-candidate+audit+archive reservation'
    old=read(previous/'protocol.json');p=copy.deepcopy(old)
    cap=read(old['capture_summary_path'])['result'];fresh=cap['samples'];retention=old['retention_rows']
    assert len(fresh)==16 and all(r['full'] and r['policy_hash']==base['model']['policy_hash'] for r in fresh)
    assert len(retention)==8 and all(r['policy_hash']==base['model']['policy_hash'] for r in retention)
    target_ids=['dev_task6_01','dev_task6_04','dev_task6_09','dev_task6_13','dev_task6_18','dev_task6_21','dev_task6_24','hold_task6_00']
    jobs={j['id']:j for j in old['old275_jobs']}
    lost=report['summaries']['pair']['result']['candidates']['B']['lost_ids']
    assert len(set(target_ids))==8 and set(target_ids)<=set(base['success_ids']) and set(target_ids)<=set(lost)
    reused=[r for r in cap['replay_rows'] if L.task(r['job_id'])!=6]
    assert len(reused)==7
    assert all(jobs[x]['case']['step_cap']==23334 for x in target_ids), '384KiB proof requires exact old5cap23334'
    p.update(id='T486_TARGETED_LEFT5_ACTUAL_ADAM_DISPLACEMENT_PROJECTION',out=str(OUT),initial=base['model'],
        initial_result=base,initial_success_ids=base['success_ids'],retention_rows=retention,
        fresh_rows=fresh,preflight_rows=old['preflight_rows'],replay_rows=cap['replay_rows'],
        replay_jobs=[jobs[x] for x in target_ids],targeted_ids=target_ids,
        seed=20261111486,epochs=16,anchor_mode='actual_Adam_projection',reused_replay_rows=reused,fresh_groups=[0]*8+[1]*8,
        max_bytes=14*2**20,metadata_reserved_bytes=6*2**20,
        full_bytes=384*1024,compact_bytes=8192,model_bytes=1792*1024,
        max_physics=12000000,max_commands=500,main_seconds=4800,learning_seconds=1200,
        audit_seconds=600,root_max_physics=12000000,root_max_commands=500,root_max_main_seconds=4800,
        all_seconds=6600,absolute_all_stop=stop,absolute_native_stop=stop-2700,
        absolute_learning_stop=stop-5400,preflight_commands=6,
        user_authorization='Nov11 user explicitly authorizes bounded background experiments until today20:30 Beijing. One new finite targeted-memory projected-Adam pilot, does not extend T480/T481.',
        mechanism='Eight new oldleft5 realhistory episodes plus seven historical otherfamily centers, taskbalanced Vtrace; project actual matureAdam Actor displacement onto eight per-left5-episode empirical surrogate halfspaces. GEM-like jointpilot, not original/causal/safetyproof; Adam moments retained, full383 mandatory.',
        acceptance='Each complete383 must preserve all335 initial successIDs and strictly increase wide8 successes. Reject/interruption entire Actor/Value/bothAdam rollback. On first accepted model stop later learning and preserve; no seed/epoch/noise/weight grid.',
        decision='T481 actual16 fresh reused only with exactly initial policyhash and full audited density/history. Eight new targeted stochastic histories+eight exact. Retention eight audited histories reused. fresh16 NAVIGATION holdout/reload78 remain unallocated null.',
        reference_semantics='Eight prelocked T485 lost oldLEFT5 IDs each one new stochastic full episode plus seven audited old-otherfamily center histories, all outcomes used. Each family equal1/8; oldLEFT5 eight shareits1/8. Eight separate LEFT5 episode gradient halfspaces, finite surrogate only. Not native safety certificate or proof of otherfamily retention.')
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={r['path'] for r in fresh+retention+p['replay_rows']}
    files|={str(previous/'protocol.json'),old['capture_summary_path'],old['capture_audit_path'],
            str(previous/'pair/summary.json'),str(previous/'pair/audit.json'),str(DOC/'T485_FINAL_REPORT.json'),
            str(DOC/'T486_PLAN.md'),str(ROOT/'stage/docs/research_20261111_wide8/T486_ACTUAL_ADAM_PROJECTION_PRIMARY.md'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T485.json')}
    p['protected_files']=dict(old['protected_files']);files|=set(p['protected_files'])
    needed=max(1,M.used()+p['max_bytes']+2**20-2**30)
    enc=S.plan(files,needed);p['storage_rows']=enc['rows'];p['storage_expected_release']=enc['selected_net_release_bytes']
    remain=max(1,needed-enc['selected_net_release_bytes']+512*1024)
    cache=K.plan(files,remain) if not enc['admitted'] else None
    if cache is None:remain=0
    for row in enc['rows']:
        for key in ['receipt','manifest']:files.add(str(ROOT/row[key]))
    for group in ([] if cache is None else cache['groups']):
        for key in ['receipt','manifest']:files.add(str(ROOT/group[key]))
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    p['cache_total_reserved_bytes']=p['max_bytes'];p['cache_hard_stop_epoch']=time.time()+1200
    p['remote_cache_required']=cache is not None;p['remote_cache_path']=str(HERE/'cache_plan.json')
    save(HERE/'cache_plan.json',cache);p['remote_cache_sha256']=sha(HERE/'cache_plan.json')
    worst=2*sum(j['case']['step_cap'] for j in p['replay_jobs'])+sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs'])
    storage=8*p['full_bytes']+383*p['compact_bytes']+p['model_bytes']+p['metadata_reserved_bytes']
    assert worst<=p['root_max_physics'] and storage<=p['max_bytes']
    assert len(json.dumps(p).encode())<2**21 and (HERE/'cache_plan.json').stat().st_size<2**21
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T486_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=p['root_max_physics'],worst_physics=worst,
        max_commands=500,worst_commands=413,own_bytes=p['max_bytes'],worst_storage=storage,global_limit=2**30,
        compression_expected_release=p['storage_expected_release'],cache_required_release=remain,
        all_stop=stop,main_seconds=4800,independent_audit_seconds=780,backup_source_seconds=600,
        max_live_Adam=32,shadow_Adam=32,independent_historical_Adam=32,preflight_discarded_Adam=8,
        targeted_ids=target_ids,fresh16=None,reload78=None))
    print('T486_REGISTERED_NOT_LAUNCHED',flush=True)


def compress():
    import t486_storage as S
    S.execute(read(OUT/'protocol.json'))
def compression_audit():
    import t486_storage as S
    S.audit(read(OUT/'protocol.json'))
def cache_protocol():
    p=read(OUT/'protocol.json');assert sha(p['remote_cache_path'])==p['remote_cache_sha256'];return dict(p,remote_cache=read(p['remote_cache_path']))
def cache():
    import t486_cache as K
    assert read(OUT/'compression_audit.json')['state']=='PASS_LOCAL_REVERSIBLE_ENCODING'
    p=read(OUT/'protocol.json')
    if p['remote_cache_required']:K.evict(cache_protocol())
    else:
        K._idle();assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
        assert not (OUT/'cache_summary.json').exists()
        save(OUT/'cache_summary.json',dict(state='NO_REMOTE_CACHE_EVICTION_NEEDED_LOCAL_REVERSIBLE_ENCODING_SUFFICIENT',
            pid=os.getpid(),files=0,native=0,optimizer=0,unbacked_unique_evidence_deleted=0))
def cache_audit():
    import t486_cache as K
    p=read(OUT/'protocol.json')
    if p['remote_cache_required']:K.cacheaudit(cache_protocol())
    else:
        K._idle();s=read(OUT/'cache_summary.json');assert not Path('/proc',str(s['pid'])).exists()
        assert s['files']==0 and not (OUT/'cache_audit.json').exists()
        for f,h in p['inputs'].items():assert sha(f)==h,f
        assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
        save(OUT/'cache_audit.json',dict(state='PASS_REFERENCE_RECEIPT_MARKERS_AND_PROTECTED_INPUTS',
            files=0,native=0,optimizer=0,no_remote_unlink_performed=True,
            full_root_reserved_after_independent_lossless_encoding_audit=True))


def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True)
    assert read(OUT/'cache_audit.json')['state']=='PASS_REFERENCE_RECEIPT_MARKERS_AND_PROTECTED_INPUTS'
    assert not (OUT/'preflight.json').exists()
    # Shape/density clone fixture, not oldtask collection; real old8 gate below.
    fixture=p['replay_rows']
    refs=L.references(p,p['retention_rows'],fixture);obj=L.make(p);shadow=L.make(p)
    data=L.memory(p,fixture,next(obj[0].parameters()).device)
    assert float(L.keep_loss(obj[0],refs,0)[0].detach())<1e-24
    four=[E.N.S.arrays(r['path']) for r in p['preflight_rows']]
    arrays=four[:2]*4+four[2:]*4
    pp=dict(p,projection_fixture_ids=[r['job_id'] for r in fixture])
    aa=L.fit(obj,arrays,pp,refs,data,lambda:None,lambda *_:None,steps=2)
    bb=L.fit(shadow,arrays,pp,refs,data,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
    results={'actual_Adam_projection':aa}
    state=P.state(*obj);buf=io.BytesIO();torch.save(dict(joint_state=state,policy_hash=obj[0].phash(),joint_hash=digest(state),architecture='plain120_128_64GRU_2',single_actor=True),buf)
    blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    assert digest(torch.load(io.BytesIO(gzip.decompress(blob)),map_location='cpu',weights_only=False)['joint_state'])==digest(state)
    for o in [obj,shadow]:P.load(o,P.checkpoint(p['initial']['path'])['joint_state']);assert digest(P.state(*o))==p['initial']['joint_hash']
    folder=OUT/'preflight_native';folder.mkdir();counts=R.M.mp.get_context('spawn').RawArray('q',3)
    pool=T.T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        reply=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['replay_jobs'][:2]])
        assert all(r['state']=='CANCEL_PASS' for r in reply) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_Adam=sum(4*r['epochs'] for r in results.values()),maximum_discarded_Adam=8,
        actual_old8_independently_audited_reused=True,fixture_history_density_exact=True,clone_result=results,
        mature_Actor_Value_Adam_shadow_rollback_exact=True,serialized_bytes=len(blob),single_pilot=True))
    print('T486_PREFLIGHT_PASS_NOT_LAUNCHED',flush=True)


def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();duration=6600;assert now+duration<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T486','--why=Bounded old-task recurrent RL rehearsal',
            'timeout','--signal=TERM','--kill-after=2',str(duration),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+duration,source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),
        offline_no_LLM_network_install=True,single_pilot_NOT_matched_ablation=True,new_window_not_T481_extension=True))
    print('T486_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
