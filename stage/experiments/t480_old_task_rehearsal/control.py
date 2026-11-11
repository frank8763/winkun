"""Next direction: one bounded old-task V-trace rehearsal pilot, not A/B."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,gzip,io,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t480_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT,DOC=T.OUT,ROOT/'stage/docs/review_20261111'


def prepare():
    import t480_cache as K
    import t480_storage as S
    assert not OUT.exists();K._idle()
    previous=ROOT/'stage/runs/t479_balanced_success_sil'
    report=read(DOC/'T479_FINAL_REPORT.json')
    assert report['remote']==K.COMPLETE
    predecessor_audit=previous/'pair/audit.json'
    if report['state']=='COMPLETE_AUDITED_LOCAL':
        assert read(predecessor_audit)['state']=='PASS'
        assert report['summaries']['pair']['result']['chosen_arm'] is None,'Successful previous proposal: stop this failed-both design'
    else:
        assert report['state']=='BOUNDED_QUEUE_STOPPED' and report['completed']==['capture']
        predecessor_audit=DOC/'T479_PARTIAL_REVIEW.json'
        review=read(predecessor_audit)
        assert review['state']=='PASS_PARTIAL_ADMISSION_REVIEW_NOT_COMPLETE_T479_AUDIT'
        assert review['new_policy_updates']==0 and review['historical_discarded_Adam']==64
        assert review['selected']==report['selected']
        assert all(c['already_lost_success_ids'] for c in review['candidates'].values())
        assert read(previous/'capture/audit.json')['state']=='PASS'
    base=report['selected'];assert base['wide8']['success']==8 and base['wide6']['success']==53 and base['old275']['success']==274
    stop=datetime(2026,11,11,16,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    assert time.time()+7200<stop
    old=read(previous/'protocol.json');p=copy.deepcopy(old)
    retention=read(previous/'capture/summary.json')['result']['success_rows']
    assert len(retention)==8 and len({r['job_id'] for r in retention})==8
    jobs={j['id']:j for j in old['old275_jobs']}
    old_ids=[f'dev_task{task}_13' for task in range(8)]
    assert set(old_ids)<=set(base['success_ids'])
    fixture=read(previous/'capture/summary.json')['result']['samples']
    assert len(fixture)==4 and all(r['full'] for r in fixture)
    p.update(id='T480_OLD_TASK_RECURRENT_VTRACE_REHEARSAL_PILOT',out=str(OUT),
        initial=base['model'],initial_result=base,initial_success_ids=base['success_ids'],
        capture_success_ids=[r['job_id'] for r in retention],retention_rows=retention,
        replay_jobs=[jobs[x] for x in old_ids],preflight_rows=fixture,
        seed=20261111480,epochs=16,replay_actor_weight=1.,replay_value_weight=1.,keep_weight=1.,
        max_bytes=20*2**20,full_bytes=800*1024,compact_bytes=8192,model_bytes=1792*1024,
        max_physics=12000000,max_commands=500,main_seconds=4500,learning_seconds=1200,
        audit_seconds=900,root_max_physics=12000000,root_max_commands=500,
        root_max_main_seconds=4500,all_seconds=6900,
        absolute_all_stop=stop,absolute_native_stop=stop-1800,absolute_learning_stop=stop-5400,
        user_authorization='Nov11 user: check then continue nextdirection. One new finite old-taskRL auxiliarypilot, not T479 restart/extraSIL budget. All by16:30Beijing, earlier relative deadlines strict.',
        mechanism='Single recurrentPPO proposal plus actual_old8task stochastic VtraceRL replay. rho/c/pg clips1, full physical finite MC semantics; both old success/failure used. Existing balanced8success and old211 function anchors plus newactualold8anchors. Coverage+RL jointpilot, NOT matchedablation/fullCLEAR/original/safetyproof.',
        acceptance='One full383, preserve all335current successIDs AND increaseeight success; otherwise entire Actor/Value/bothAdam rollback. No second round/no A/no seed/coefficient/grid or deadline extension.',
        decision='Own8 successfulls reused from auditedT479, not new independent data. New8old stochastic+4eight fresh and12exact, no teacher. Fresh16/reload78 unallocated null; no stableeight claim.',
        reference_semantics='Eight oldtasks fixed center13, one iidnoise episode each. All outcomes used as true stochastic replay with behavior atoms/logp/fullhistory, never deterministic evaluation labeledasGaussian. New4eight batch freshonpolicy separate. Every finite terminal including true cap zero bootstrap; no administrative censor admitted.',
        metadata_reserved_bytes=5*2**20)
    # Preserve the simulator's complete historical constructor contracts. No
    # historical input is pruned merely to gain storage admission.
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={r['path'] for r in retention+fixture}
    files|={str(previous/'protocol.json'),str(previous/'capture/summary.json'),
            str(previous/'capture/audit.json'),str(predecessor_audit),
            str(DOC/'T479_FINAL_REPORT.json'),str(DOC/'T480_PLAN.md'),
            str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T479.json')}
    p['protected_files']=dict(old['protected_files']);files|=set(p['protected_files'])
    needed=max(1,M.used()+p['max_bytes']+2**20-2**30)
    enc=S.plan(files,needed)
    p['storage_rows']=enc['rows'];p['storage_expected_release']=enc['selected_net_release_bytes']
    # Compression stage keeps all original bytes locally. Full root admission
    # is checked after this and optional remote-verified NPZ cache eviction.
    remain=max(1,needed-enc['selected_net_release_bytes']+512*1024)
    cache=K.plan(files,remain) if not enc['admitted'] else None
    if cache is None:remain=0
    for row in enc['rows']:
        for key in ['receipt','manifest']:files.add(str(ROOT/row[key]))
    for group in ([] if cache is None else cache['groups']):
        for key in ['receipt','manifest']:files.add(str(ROOT/group[key]))
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    p['cache_total_reserved_bytes']=p['max_bytes'];p['cache_hard_stop_epoch']=time.time()+1200
    p['remote_cache_required']=cache is not None
    p['remote_cache_path']=str(HERE/'cache_plan.json')
    (HERE/'cache_plan.json').write_text(json.dumps(cache,ensure_ascii=False,separators=(',',':'))+'\n')
    p['remote_cache_sha256']=sha(HERE/'cache_plan.json')
    worst=2*sum(j['case']['step_cap'] for j in p['replay_jobs']+p['training_jobs'])+sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs'])
    storage=12*p['full_bytes']+383*p['compact_bytes']+p['model_bytes']+p['metadata_reserved_bytes']
    assert worst<=p['root_max_physics'] and storage<=p['max_bytes']
    assert len(json.dumps(p).encode())<2**21 and (HERE/'cache_plan.json').stat().st_size<2**21
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T480_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=12000000,worst_physics=worst,
        max_commands=500,worst_commands=421,own_bytes=p['max_bytes'],worst_storage=storage,global_limit=2**30,
        compression_expected_release=p['storage_expected_release'],cache_required_release=remain,
        all_stop=stop,main_seconds=4500,independent_audit_seconds=1200,backup_source_seconds=900,
        max_live_Adam=32,shadow_Adam=32,independent_historical_Adam=32,preflight_discarded_Adam=8,
        fresh16=None,reload78=None))
    print('T480_REGISTERED_NOT_LAUNCHED',flush=True)


def compress():
    import t480_storage as S
    S.execute(read(OUT/'protocol.json'))
def compression_audit():
    import t480_storage as S
    S.audit(read(OUT/'protocol.json'))
def cache_protocol():
    p=read(OUT/'protocol.json');assert sha(p['remote_cache_path'])==p['remote_cache_sha256'];return dict(p,remote_cache=read(p['remote_cache_path']))
def cache():
    import t480_cache as K
    assert read(OUT/'compression_audit.json')['state']=='PASS_LOCAL_REVERSIBLE_ENCODING'
    p=read(OUT/'protocol.json')
    if p['remote_cache_required']:K.evict(cache_protocol())
    else:
        K._idle();assert M.used()-M.used(OUT)+p['max_bytes']<=2**30
        assert not (OUT/'cache_summary.json').exists()
        save(OUT/'cache_summary.json',dict(state='NO_REMOTE_CACHE_EVICTION_NEEDED_LOCAL_REVERSIBLE_ENCODING_SUFFICIENT',
            pid=os.getpid(),files=0,native=0,optimizer=0,unbacked_unique_evidence_deleted=0))
def cache_audit():
    import t480_cache as K
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
    fixture=[dict(p['preflight_rows'][i%4],job_id=p['replay_jobs'][i]['id']) for i in range(8)]
    refs=L.references(p,p['retention_rows'],fixture);obj=L.make(p);shadow=L.make(p)
    data=L.memory(p,fixture,next(obj[0].parameters()).device)
    assert float(L.keep_loss(obj[0],refs,0)[0].detach())<1e-24
    arrays=[E.N.S.arrays(r['path']) for r in p['preflight_rows']]
    aa=L.fit(obj,arrays,p,refs,data,lambda:None,lambda *_:None,steps=2)
    bb=L.fit(shadow,arrays,p,refs,data,lambda:None,lambda *_:None,steps=2)
    assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
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
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_Adam=4*aa['epochs'],maximum_discarded_Adam=8,
        actual_old8_NOT_yet_collected=True,fixture_history_density_exact=True,clone_result=aa,
        mature_Actor_Value_Adam_shadow_rollback_exact=True,serialized_bytes=len(blob),single_pilot=True))
    print('T480_PREFLIGHT_PASS_NOT_LAUNCHED',flush=True)


def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+6900<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T480','--why=Bounded old-task recurrent RL rehearsal',
            'timeout','--signal=TERM','--kill-after=2','6900','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+6900,source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),
        offline_no_LLM_network_install=True,single_pilot_NOT_matched_ablation=True,new_window_not_T479_extension=True))
    print('T480_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
