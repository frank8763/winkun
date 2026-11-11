"""New bounded morning window; exact storage reserve precedes any simulation."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,copy,io,gzip,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t478_runtime as T
L,B,P,E,M,R,torch,np=T.L,T.B,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT=T.OUT;DOC=ROOT/'stage/docs/review_20261111'

def prepare():
    import t478_storage as S
    assert not OUT.exists();src=ROOT/'stage/runs/t477_wide8_recovery';receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T477.json';rc=read(receipt);assert rc['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';assert all(not Path('/proc',str(pid)).exists() for pid in read(src/'SEALED.json')['scientific_pids'])
    old=read(src/'protocol.json');ss=read(src/'longcredit/summary.json');assert read(src/'longcredit/audit.json')['state']=='PASS';base=ss['candidate'];assert base['wide8']['success']==8 and base['wide6']['success']==53 and base['old275']['success']==274
    raw=[json.loads(x) for x in (src/'longcredit/cases.jsonl').read_text().splitlines()];final={r['job_id']:r for r in raw if r['role']=='FINAL_ALL383'};p=copy.deepcopy(old)
    stop=datetime(2026,11,11,12,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+10500<stop
    probeids=['wide117_8m_dev6_RIGHT_13','wide117_8m_dev6_RIGHT_23','wide117_8m_dev6_LEFT_01','wide117_80_dev6_LEFT_00','wide117_80_dev6_LEFT_13','wide117_80_dev6_RIGHT_13'];jobs={j['id']:j for j in old['wide8_jobs']+old['six_jobs']+old['old275_jobs']}
    original=read(ROOT/'stage/runs/t449_offline_joint_recovery/protocol.json')
    own=[r for r in raw if r['role'] in ['TRAIN_1','TRAIN_2'] and r['job_id']=='wide117_8m_dev6_RIGHT_13'];fixture=[next(r for r in raw if r['role']==f'TRAIN_{rnd}' and r['job_id']==f'wide117_8m_dev6_{side}_13') for side in ['LEFT','RIGHT'] for rnd in [1,2]]
    p.update(id='T478_WIDE8_POOLED_CREDIT_AND_FULL_RETAINED_SET',out=str(OUT),seed=20261111478,initial=base['model'],initial_result=base,initial_success_ids=base['success_ids'],old_states=original['old_states'],own_reference_rows=own,preflight_rows=fixture,
        probe_jobs=[jobs[x] for x in probeids],probe_cached={x:final[x] for x in probeids},probe_success_reference_ids=probeids[-2:],training_jobs=[jobs[x] for x in ['wide117_8m_dev6_LEFT_13','wide117_8m_dev6_LEFT_01','wide117_8m_dev6_RIGHT_13','wide117_8m_dev6_RIGHT_23']],
        max_bytes=23*2**20,full_bytes=1024*1024,compact_bytes=8192,model_bytes=1792*1024,max_physics=23000000,max_commands=850,main_seconds=7200,learning_seconds=4800,audit_seconds=1200,epochs=16,rounds=2,lambda_native=1.,keep_weight=1.,
        root_max_physics=24000000,root_max_commands=1000,root_max_main_seconds=8100,all_seconds=10800,
        absolute_all_stop=stop,absolute_learning_stop=stop-3600,absolute_native_stop=stop-1800,
        user_authorization='Nov11 usercontinueget8m,analyzeandsolve: new boundedmorningwindow, not resume or extend closed09:00night. All by12:30Beijing,26MNOTauthorized, actual24Mhardlimit/1000commands.',
        acceptance='Everycandidate full383: all335initial/current successIDs mustsurvive andeight successes muststrictlyincrease. Otherwise entireActor,V,twoAdam rollback. No12panelonly acceptance. Atmost2rounds,2consecutive rejectsstop.',
        mechanism='Completeepisode lambda1 recurrentPPO; pooled within-side mean/std across2own-policy episodes eachside (4batch), initialfunction preservation on8rotatingoldcontexts andactualsuccessfulcontexts, coefficient1. Preserve currentmatureActor/V/Adam. Jointengineeringpilot, notsinglefactor causalororiginalmethod.',
        task='Exactly same T4778m1.17mwide53334native106.668s, samevehicle/observations/reward/sigma/maxspeed/.2mGOAL; no cap/radius relaxation, no map/planner/waypoint/teacher/router.',
        reference_semantics='Old211 and successfulactualstates are finitefunction-preservation contexts with frozeninitialunclippedlocations, not provenrecoveryexperts or newonpolicyPPO trajectories. Only4freshcurrentpolicyepisodes perround enterPPO.',
        decision='Researchdevelopmentonly; no16fresh/78modelreloadallocated. Full383acceptance cannotproveunseenstability. Capacitymayadmit1roundonly; do not add seeds/epochs/newid to extend.',
        inherited_failures='old dev_task6_00 contact, six wide117_80_dev6_LEFT_00 timeout, twooriginalbaseline8successlost. Notfixed byclaim.',
        renewed_window_not_expanding_old_budget=True,root_metadata_reservation=2*2**20)
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T478_PLAN.md'),str(src/'protocol.json'),str(src/'longcredit/summary.json'),str(src/'longcredit/audit.json'),str(receipt),base['model']['path'],p['old_states']}
    files|={r['path'] for r in own+fixture};p['protected_files']={**old['protected_files'],base['model']['path']:base['model']['sha256'],p['old_states']:sha(p['old_states'])}
    need=max(0,M.used()+p['max_bytes']+2*2**20-2**30);encoding=S.plan(files,need);assert encoding['admitted'],('Notenoughlosslessbackedlogs',encoding['available_net_release_bytes'],need)
    p['storage_rows']=encoding['rows'];p['storage_expected_release']=encoding['selected_net_release_bytes']
    for row in encoding['rows']:
        for key in ['receipt','manifest']:files.add(str(ROOT/row[key]))
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    worst=2*sum(j['case']['step_cap'] for j in p['probe_jobs'])+2*(2*sum(j['case']['step_cap'] for j in p['training_jobs'])+sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs']))
    assert worst<=24000000;save(DOC/'T478_BUDGET.json',dict(state='REGISTERED_BEFORE_STORAGE_PREFLIGHT_LAUNCH',max_physics=24000000,worst_physics=worst,max_commands=1000,worst_commands=16+2*(8+383)+4+8,own_reserved_bytes=p['max_bytes'],before_logical=M.used(),lossless_expected_release=encoding['selected_net_release_bytes'],after_full_reserved=M.used()-M.used(OUT)-encoding['selected_net_release_bytes']+p['max_bytes'],global_limit=2**30,all_stop_epoch=stop,old_night_not_resumed=True,full2rounds_NOT_guaranteed_capacity_gate=True));print('T478_REGISTERED',flush=True)

def compress():
    import t478_storage as S
    p=read(OUT/'protocol.json');S.execute(p)
def compression_audit():
    import t478_storage as S
    p=read(OUT/'protocol.json');S.audit(p);T.verify(p,True)

def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'compression_audit.json')['state']=='PASS_LOCAL_REVERSIBLE_ENCODING' and not (OUT/'preflight.json').exists();refs=L.references(p);obj=L.make(p);shadow=L.make(p);before=P.state(*obj);initial_loss=L.keep_loss(obj[0],refs,0)
    assert float(initial_loss[0].detach())<1e-24
    arrays=[E.N.S.arrays(r['path']) for r in p['preflight_rows']];a=L.fit(obj,arrays,p,refs,0,lambda:None,lambda *_:None,steps=1);b=L.fit(shadow,arrays,p,refs,0,lambda:None,lambda *_:None,steps=1);assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
    buf=io.BytesIO();state=P.state(*obj);torch.save(dict(joint_state=state,policy_hash=obj[0].phash(),joint_hash=digest(state),architecture='plain120_128_64GRU_2',single_actor=True),buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];assert digest(torch.load(io.BytesIO(gzip.decompress(blob)),map_location='cpu',weights_only=False)['joint_state'])==digest(state)
    P.load(obj,before);P.load(shadow,before);assert digest(P.state(*obj))==p['initial']['joint_hash']==digest(P.state(*shadow));assert any(abs(x['normalized_mean'])>1e-5 for x in a['pooled_advantage'])
    for group in [0,1]:assert abs(np.mean([x['normalized_mean'] for x in a['pooled_advantage'] if x['group']==group]))<1e-10
    # Same old12panel counterexample: currentactualretained set detects the real6m regression.
    parent=read(ROOT/'stage/runs/t449_offline_joint_recovery/summary.json');oldbest=next(v['candidate'] for v in parent['phases'] if v['phase']=='recovery');assert 'wide117_80_dev6_LEFT_00' in oldbest['success_ids'] and 'wide117_80_dev6_LEFT_00' not in p['initial_success_ids']
    assert len(p['initial_success_ids'])==335
    folder=OUT/'preflight_native';folder.mkdir();counts=R.M.mp.get_context('spawn').RawArray('q',3);pool=T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:replies=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['probe_jobs'][:2]]);assert all(v['state']=='CANCEL_PASS' for v in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=6,discarded_synthetic_or_historical_clone_Actor_Adam=2,discarded_clone_Value_Adam=2,real4history_density_and_hidden_contract=True,all335acceptance_regression_fixture=True,initial_reference_loss=float(initial_loss[0].detach()),mature_serialized_bytes=len(blob),actual_update=a,all_Actor_Value_Adam_shadow_and_rollback_exact=True,new_contact_reporting_constructed_before_first_native=True));print('T478_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],text=True).strip();now=time.time();assert now+10200<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T478','--why=Boundedwide8morningrepair,all12:30stop','timeout','--signal=TERM','--kill-after=2',str(min(10800,int(p['absolute_all_stop']-now))),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=min(now+10800,p['absolute_all_stop']),source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),new_morning_window_not_night_restart=True,offline_no_LLM_network_install=True));print('T478_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
