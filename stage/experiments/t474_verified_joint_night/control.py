"""Register a finite, offline, verified-recovery night relay."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t474_verified_joint_night';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import t474_runtime as T
L,M,E,R,torch,np=T.L,T.M,T.E,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest

def prepare():
    assert not OUT.exists(),'Neverrestart'
    olddir=ROOT/'stage/runs/t473_full_horizon_teacher';raw=read(olddir/'protocol.json');last=read(olddir/'summary.json')
    assert last['state']=='FAILED' and read(DOC/'T473_FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(v)).exists() for v in read(olddir/'SEALED.json')['scientific_pids'])
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(olddir/f)['pid'])).exists()
    ref=last['result'];assert ref['success'] and not ref['collision'] and ref['terminal_native']==21205
    prior=ROOT/'stage/runs/t468_offline_recovery_night';night=read(prior/'protocol.json');initial=night['initial_candidate'];science=copy.deepcopy(night['science']);s469=read(ROOT/'stage/runs/t469_counterexample_recovery/summary.json');s471=read(ROOT/'stage/runs/t471_reset_recovery/summary.json')
    corrections=[read(prior/'occupancy_0/summary.json')['recovery_rows'][0]]
    # Explicitly choose00 and21 byID, not serialization order.
    rows=read(prior/'occupancy_0/summary.json')['recovery_rows'];corrections=[next(v for v in rows if v['job_id']==k) for k in ['dev_task6_00','dev_task6_21']]
    reset03=next(v for v in s471['query_rows'] if v['job_id']=='dev_task6_03');assert reset03['success']
    science.update(learning_initial=initial['model'],current_success_ids=initial['success_ids'],counterexample_model=read(prior/'occupancy_0/summary.json')['proposal'],
        seed_occupancy_rows=night['seed_occupancy_rows'],baseline_success_rows=s469['baseline_rows'],original_corrections=corrections,new_reset03=reset03,
        wide_counterexample_student_rows=[v for v in s469['student_rows'] if v['job_id'].startswith('wide')],wide_counterexample_recovery_rows=[v for v in s469['query_rows'] if v['job_id'].startswith('wide')],
        raw15_reference=ref,verified_raw15=ref,expected_reset_teacher_digest=reset03['teacher_source_SHA256'],recovery_teacher_model=night['science']['recovery_teacher_model'],protected_files=night['protected_files'])
    p=dict(id='T474_VERIFIED_RAW_RECOVERY_JOINT_NIGHT',out=str(OUT),science=science,raw_science=raw,initial_candidate=initial,modes=['joint','priority'],
        max_bytes=64*2**20,root_metadata_bytes=4*2**20,verification_partition_bytes=3*2**20,train_partition_bytes=10*2**20,gate_partition_bytes=43*2**20,
        max_physics=20000000,max_commands=900,max_main_seconds=7800,preflight_commands=26,
        night_prior_physics=raw['night_prior_physics']+last['physics'],night_prior_commands=raw['night_prior_commands']+last['commands'],night_prior_main_seconds=raw['night_prior_main_seconds']+last['elapsed_s'],
        night_max_physics=40000000,night_max_commands=2000,night_max_main_seconds=14400,
        absolute_learning_stop=night['absolute_learning_stop'],absolute_native_stop=night['absolute_native_stop'],absolute_all_stop=night['absolute_all_stop'],
        raw15_reference=ref,protected_files=night['protected_files'],
        protocol='Frozen success control only: noCEM/search rerun. Twoindependent full originalstartnative exactreplays+audit gate beforetraining. T473FAILED preserved. Metadata regressionexpectfirst_search_passdifference and rejectarraydigestchange.',
        training='Twofixed128live128shadow branches, sameoriginal327init; joint ordinarycorrection+preservation sum, priority projectsconflictingcorrectiongradientagainstfinitepreservationgradient beforeoneactualmatureAdamstep. Adampreconditioning meansnotglobalnoninterferenceguarantee.',
        labels='4heterogeneousoffpolicyrecoveries;00/21/03 neuralteacherlabels withfirst+1early1tail.1;raw15onlyexecutedownershipsuffixislabel weight1, nofictionalexpertlabelsinprefix.4trajectoryequalweights.Originalreference/own+widecounterexamplefunctions preserved.',
        transfer='OnlypreserveALLcurrent successes and increasecount afteraudit may bekept. Eachbranch independently initializesoriginal327 (not previousrejectedproposal). If329allsafe stopalltraining andrun16fresh/78independentreadback; noadaptationafterfresh.',
        admission='Afteractualsealedhistory, nextwhole10MiBtraining+43MiBgate+1MiBmetadata mustfit64; noexpansion or deletion. Atmosttwofixedbranches, noadditionalepochs,seeds,noiseorgrid.',
        offline='Scientificchain requiresnoLLM/network/dependencyinstall; eachstageaudit andlocalmanifest beforehandover. RemotePENDING maywaituntilnetworkreturns; networkusedonlyafterallscienceends.',
        stop='Any source/array/model/rollback/density/audit failure stopschain; verificationfail stopsalltraining. Finitequeueendswhencomplete, neverartificiallykeeprunningtillmorning.',
        user_authorization='Nov10 user: do not repeat mistakes; starttonightofflineunattended. Continue sameconfirmedNov11 08:30hardstop andexisting40M/2000/14400nightbudget. Userreviews6mbefore8.',eight_started=False,eight_implemented=False)
    # Root already locks all raw inputs; do not duplicate that map in child science.
    for key in ['inputs','constructor_reads']:p['raw_science'].pop(key,None)
    p['worst_physics']=46668+2*sum(j['case']['step_cap'] for j in science['main_jobs']+science['old275_jobs'])+4000000
    p['worst_commands']=26+3+2*345+126
    assert p['worst_physics']<=p['max_physics'] and p['worst_commands']<=p['max_commands']
    assert p['night_prior_physics']+p['max_physics']<=p['night_max_physics'] and p['night_prior_commands']+p['max_commands']<=p['night_max_commands']
    assert p['night_prior_main_seconds']+p['max_main_seconds']<=p['night_max_main_seconds']
    assert M.used()+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()+600+300+2400+600+1800+600+900<p['absolute_all_stop']
    files=set(raw['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T474_PLAN.md'),str(DOC/'T473_FINAL_REPORT.json'),str(olddir/'summary.json'),str(olddir/'failure.json'),str(olddir/'SEALED.json'),str(olddir/'selected.json'),str(olddir/'branches.jsonl'),ref['path'],str(olddir/'partial.npz'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T473.json')}
    files|={v['path'] for v in corrections+[reset03]+s469['baseline_rows']+science['wide_counterexample_student_rows']+science['wide_counterexample_recovery_rows']}
    p['inputs']={f:sha(f) for f in sorted(files)};OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    save(DOC/'T474_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_bytes=p['max_bytes'],fully_reserved_global=M.used()-M.used(OUT)+p['max_bytes'],worst_physics=p['worst_physics'],max_physics=p['max_physics'],worst_commands=p['worst_commands'],max_commands=p['max_commands'],night_prior_physics=p['night_prior_physics'],night_prior_commands=p['night_prior_commands'],night_max_cumulative_physics=p['night_prior_physics']+p['max_physics'],night_max_cumulative_commands=p['night_prior_commands']+p['max_commands'],max_main_seconds=p['max_main_seconds'],all_stop=p['absolute_all_stop'],bothbranchesconditionalactualstorageadmission=True))
    print('T474_REGISTERED',flush=True)

def train_config(root,mode,rawrow):
    p=copy.deepcopy(root['science']);p.update(id='T474_'+mode.upper(),out=str(OUT/mode),mode=mode,verified_raw15=rawrow,
        max_bytes=root['train_partition_bytes'],max_physics=8000000,max_commands=400,preflight_commands=0,main_seconds=2400,learning_seconds=1200,model_bytes=768*1024,full_bytes=448*1024,compact_bytes=4096,
        absolute_learning_stop=root['absolute_learning_stop'],absolute_native_stop=root['absolute_native_stop'],absolute_all_stop=root['absolute_all_stop'])
    return p

def verify_config(root):
    p=copy.deepcopy(root['raw_science']);p.update(out=str(OUT/'verified'),max_bytes=root['verification_partition_bytes'],max_physics=50000,max_commands=3,main_seconds=600,audit_seconds=300,raw15_reference=root['raw15_reference'])
    return p

def preflight():
    root=read(OUT/'protocol.json');T.verify(root,True);assert not (OUT/'preflight.json').exists()
    old=[json.loads(x) for x in (ROOT/'stage/runs/t473_full_horizon_teacher/branches.jsonl').read_text().splitlines()][0]
    repeat=copy.deepcopy(old);repeat.update(pass_index=1,first_search_pass=False);assert T.records_equal(old,repeat)
    broken=copy.deepcopy(repeat);broken['array_digest']='CORRUPT';assert not T.records_equal(old,broken)
    result={}
    for mode in L.MODES:
        p=train_config(root,mode,root['raw15_reference']);a,b=L.make(p),L.make(p);initial=L.state(*a);batch=L.data(p);va=L.bc_step(a,batch);vb=L.bc_step(b,batch)
        assert va==vb and digest(L.state(*a))==digest(L.state(*b))
        assert va['reference218']<1e-24 and va['own10']<1e-24 and va['counterexample_function']==0
        model=L.export(a,OUT/f'preflight_{mode}.pt.gz',p['model_bytes']);assert L.frozen(p,model['path']).phash()==a[0].phash()
        L.load(a,initial);L.load(b,initial);assert digest(L.state(*a))==digest(initial)==digest(L.state(*b))
        result[mode]=va
    # Synthetic vector regression: priority removes only the conflict component.
    c=torch.tensor([-2.,3.]);g=torch.tensor([1.,0.]);projected=c-torch.dot(c,g)/torch.dot(g,g)*g
    assert torch.dot(projected,g)==0 and projected[1]==c[1]
    p=train_config(root,'joint',root['raw15_reference']);counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    pool=T.Pool(dict(p,out=str(folder),max_physics=1,native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        values=pool.request([dict(model=p['learning_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']]);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    # Actual zero-step native snapshot path, counted as two engineering commands.
    vp=verify_config(root);vf=OUT/'preflight_snapshot';vf.mkdir();T.S.OUT=vf
    session=witness=None
    with (vf/'console.log').open('w') as log,T.contextlib.redirect_stdout(log),T.contextlib.redirect_stderr(log):
        try:
            _,native,create,witness=T.S.setup(vp,vf/'native')
            def cancel(physical=False):
                if physical:raise E.L.StageStop('T474 zero-native snapshot cancellation')
            session=T.S.session(vp,native,create,witness,cancel,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')))
            snap=session.capture();before=T.S.snapshot_hash(session);session.restore(snap);assert before==T.S.snapshot_hash(session)
            with torch.no_grad():
                try:session.step();raise AssertionError('Cancellation missing')
                except E.L.StageStop:pass
            assert session.steps==0 and not session.rows
        finally:
            if session:session.close()
            if witness:witness.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=26,discarded_Adam=4,metadata_pass_field_regression_and_corruption_rejection=True,mature_ActorAdam_shadow_serialization_rollback=True,initial_all_function_preservation_zero=True,raw15_no_prefix_expert_labels=True,projection_synthetic_not_safety_certificate=True,eight_workers_cancelled=True,steps=result,verified_source_still_requires_new_independent_native_replays=True))
    print('T474_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+600+300+2400+600+1800+600+900<p['absolute_all_stop'] and now+600+300+2400+600+1800<p['absolute_native_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T474','--why=Offline verified recovery night relay until08:30',
            'timeout','--signal=TERM','--kill-after=2',str(int(p['absolute_all_stop']-now)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=p['absolute_all_stop'],learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),offline_no_network_needed=True))
    print('T474_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
