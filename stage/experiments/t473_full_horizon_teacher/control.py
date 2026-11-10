"""One bounded full-horizon native control teacher, without Actor updates."""
from pathlib import Path
import copy,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t473_full_horizon_teacher';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
from importlib.util import spec_from_file_location,module_from_spec
_spec=spec_from_file_location("stage473_control_science",HERE/"science.py")
T=module_from_spec(_spec);_spec.loader.exec_module(T)
read,save,sha=T.read,T.save,T.sha

def prepare():
    assert not OUT.exists(),'Neverrestart'
    prior=ROOT/'stage/runs/t471_reset_recovery';old=read(prior/'protocol.json');s=read(prior/'summary.json');report=read(DOC/'T471_FINAL_REPORT.json')
    assert s['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(prior/f)['pid'])).exists()
    assert all(not Path('/proc',str(v)).exists() for v in read(prior/'SEALED.json')['scientific_pids'])
    actual={r['job_id']:r for r in s['query_rows']};assert actual['dev_task6_03']['success'] and actual['dev_task6_15']['collision']
    source=next(r for r in old['source_rows'] if r['job_id']=='dev_task6_15');job=copy.deepcopy(source['request']['job'])
    baseline=ROOT/'stage/runs/t255_overnight/episodes/final_s2_reset_dev_task6_15.npz'
    p=copy.deepcopy(old)
    p.update(id='T473_FULL_REMAINING_PHASEWARP_CEM_TEACHER',out=str(OUT),job=job,source_row=source,prior_controls_path=str(baseline),prior_controls_sha256=sha(baseline),
        prefix_holds=20,teacher_first=20,remaining_holds=(job['case']['step_cap']-2000+99)//100,
        seed=20261110473,generations=3,population=12,elite=3,alpha=.5,initial_sd=[.12,1.],minimum_sd=[.03,.2],knots=16,
        max_bytes=32*2**20,max_physics=1600000,max_commands=80,max_main_commands=77,preflight_commands=2,
        full_bytes=448*1024,compact_bytes=4096,model_bytes=0,main_seconds=3000,learning_seconds=1,audit_seconds=900,backup_seconds=900,source_sync_seconds=180,all_seconds=5100,
        night_prior_physics=old['night_prior_physics']+s['physics'],night_prior_commands=old['night_prior_commands']+s['commands'],night_prior_main_seconds=old['night_prior_main_seconds']+s['elapsed_s'],
        mechanism='Fullremaining42.668s physicalcontrol CEM with16knots drive-residual/monotonephasewarp and fixed successfulT255actionprior. Earlystop atfirstnative safeGOAL, otherwise max3x12; eachfullsearch+selectedtrajectory independently repeated. Not neuralteacher replacement or takeover-timegrid.',
        proposal='controls use originalT391 clock=20+axis+5clip(phase_delta,-4,4), monotoneaccumulate, cliptopriorlast; linearinterpolatepriorfullactions plusdrive_delta, finallycommandclip[-1,1]. Zero theta follows nominal successfulprior timeline, last control held through originalcap.',
        selection='Lex actualsafeGOAL/arrivalnative/smooth, thennocontact/remaining/smooth, thencollision/remaining/smooth; fixedcandidateidties. Firstsafe immediatelystopssearch, selected is an actuallytestedfullsequence, neverunvalidatedmean.',
        supervision='Training-only privileged native model/complete-state branching. RejectedA325 is fixedprefix source/shadowActor only; sourceweights/Adamnotpromoted. Search parameters are controltrajectory variables, notActor/Adam updates or onpolicyRL data.',
        exact_contract='Bothpasses startfromoriginalA autonomous20prefix; 203nativeintegration+history/Witness/clock/previous/hidden/Actorinternalstatessnapshotrestoreexact; everycandidatefullpayload/reward/density/trajectoryhash matchessecondpass; actualselectedreplayactualnative.',
        user_authorization='Nov10 user continue afterT469/T471. Distinctfullremainingtrajectoryoptimizationtrainingteacher, not restartorgridofendedneuralteachers. Existinguserauthorizationtrainingsimteachers/deploymentonlyActor.',
        scientific_stop='Onefixedprior/phasewarpspace/seed/noise/generations only. No safecontrol meanscomplete scientificfailure,0studentBC,noextraCEM/time/seed. Source/numerical/contractsfailpreservepartialstop.',
        next_learning='Only actualteacher old15safe+nativeexact+independentaudit+fullbackup may support a separatelyregistered jointstudentpilot. NoautomaticActortraining inT473, T470/T472remainunstarted.',
        Actor_Adam=0,Value_Adam=0,new_optimizer=0,fresh16=None,reload78=None,six_gate=False,eight_started=False)
    assert job['case']['step_cap']==23334 and p['remaining_holds']==214
    p['worst_physics']=2*(2000+37*(23334-2000));assert p['worst_physics']==1582716<=p['max_physics']
    p['worst_all_commands']=79
    assert p['night_prior_physics']+p['worst_physics']<=p['night_total_max_physics'] and p['night_prior_commands']+79<=p['night_total_max_commands']
    assert p['night_prior_main_seconds']+p['main_seconds']<=p['night_total_max_main_seconds']
    assert time.time()+p['all_seconds']<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop']
    assert T.M.used()+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T473_PLAN.md'),str(DOC/'T471_FINAL_REPORT.json'),str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'SEALED.json'),str(prior/'cases.jsonl'),str(prior/'replays.jsonl'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T471.json'),str(baseline),source['path'],str(ROOT/'stage/experiments/t391_actual_student_recovery/teacher.py'),str(ROOT/'stage/experiments/t421_terminal_capture/science.py')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    save(DOC/'T473_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max_bytes=p['max_bytes'],fully_reserved_global=T.M.used()-T.M.used(OUT)+p['max_bytes'],max_physics=p['max_physics'],worst_physics=p['worst_physics'],max_commands=p['max_commands'],worst_main_commands77=True,preflight_commands2=True,worst_all_commands79=True,
        night_prior_physics=p['night_prior_physics'],night_prior_commands=p['night_prior_commands'],night_worst_physics=p['night_prior_physics']+p['worst_physics'],night_worst_commands=p['night_prior_commands']+79,
        main_seconds=3000,audit_seconds=900,backup_seconds=900,source_seconds=180,all_seconds=5100,Actor_Adam=0,eight_started=False))
    print('T473_REGISTERED',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+p['all_seconds']<p['absolute_all_stop'] and now+p['main_seconds']<p['absolute_native_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T473','--why=Bounded fullremaining native teacher, no deploymentplanner',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T473_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
