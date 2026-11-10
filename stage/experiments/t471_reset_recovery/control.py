"""Register one matched frozen reset-teacher diagnostic, not a training retry."""
from pathlib import Path
import copy,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t471_reset_recovery';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import run as T
read,save,sha=T.read,T.save,T.sha
IDS=['dev_task6_03','dev_task6_15']

def prepare():
    assert not OUT.exists(),'Neverrestart'
    prior=ROOT/'stage/runs/t469_counterexample_recovery';old=read(prior/'protocol.json');s=read(prior/'summary.json');report=read(DOC/'T469_FINAL_REPORT.json')
    assert s['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not s['all_safe']
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(prior/f)['pid'])).exists()
    assert all(not Path('/proc',str(v)).exists() for v in read(prior/'SEALED.json')['scientific_pids'])
    source={r['job_id']:r for r in s['student_rows']};queries={r['job_id']:r for r in s['query_rows']};jobs={j['id']:j for j in old['diagnostic_jobs']}
    assert all(queries[k]['collision'] and queries[k]['teacher_first']==20 and not source[k]['success'] for k in IDS)
    p=copy.deepcopy(old)
    p.update(id='T471_MATCHED_OLD_TASK_RESET_TEACHER',out=str(OUT),source_rows=[source[k] for k in IDS],prior_failed_queries=[queries[k] for k in IDS],diagnostic_jobs=[dict(jobs[k],teacher_first=20) for k in IDS],
        max_bytes=8*2**20,max_physics=100000,max_commands=48,preflight_commands=24,registered_actual_commands=44,
        full_bytes=448*1024,compact_bytes=4096,main_seconds=600,learning_seconds=1,audit_seconds=300,backup_seconds=600,source_sync_seconds=180,all_seconds=1800,teacher_first=20,
        diagnostic_reference_student={k:old['diagnostic_reference_student'][k] for k in IDS},
        diagnostic_reference_teacher={k:old['diagnostic_reference_teacher'][k] for k in IDS},
        night_prior_physics=old['night_prior_physics']+s['physics'],night_prior_commands=old['night_prior_commands']+s['commands'],night_prior_main_seconds=old['night_prior_main_seconds']+s['elapsed_s'],
        mechanism='Same rejectedA325 model and exact4sec actualprefix; change fixed training-onlyteacher from original327 recurrentcarry to safeT255reset.base_actions(float32packet). Originalclock/history/goal/caps/native/studentcarry preserved. Teacherweights and execution jointly changed, not hidden-only causal.',
        teacher_status='safeT255 historicalold275/275 is candidate recoveryteacher only; matched2native safe+exact required. No alternatepoints, seeds, weights, or teachergrid.',
        user_authorization='Nov10 user continue afterT469 report. Independent onefixedresetbaseline diagnostic for2failed recoveries, not restartT469 or extendclosedT468 mechanisms.',
        next_learning='T470 original327-function-on-fourrecoveries plan remains blocked byT469failedold2. Only newfixedteacher2safe+exact+audit/fullbackup permit separately registered changed supervised objective with explicit heterogeneousteacher labels; no automatic currentT470 launch.',
        rejected_model_scope='A325 frozen diagnostic data source only, never promoted or newstudent initialization. Lateroffpolicy data use must be registered explicitly.',
        Actor_Adam=0,Value_Adam=0,new_optimizer=0,fresh16=None,reload78=None,six_gate=False,eight_started=False)
    p['worst_physics']=2*sum(j['case']['step_cap'] for j in p['diagnostic_jobs']);assert p['worst_physics']<=p['max_physics']==100000
    assert p['night_prior_physics']+p['worst_physics']<=p['night_total_max_physics'] and p['night_prior_commands']+44<=p['night_total_max_commands']
    assert p['night_prior_main_seconds']+p['main_seconds']<=p['night_total_max_main_seconds']
    assert time.time()+p['all_seconds']<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop']
    assert T.M.used()+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|{r['path'] for r in p['source_rows']+p['prior_failed_queries']}
    files|={str(DOC/'T471_PLAN.md'),str(DOC/'T471_RECOVERY_SOURCE_REVIEW.md'),str(DOC/'T469_FINAL_REPORT.json'),str(prior/'protocol.json'),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'cases.jsonl'),str(prior/'replays.jsonl'),str(prior/'SEALED.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T469.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    save(DOC/'T471_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max_bytes=p['max_bytes'],fully_reserved_global=T.M.used()-T.M.used(OUT)+p['max_bytes'],worst_physics=p['worst_physics'],max_physics=p['max_physics'],actual_commands44=True,guard48_NOT_extra_requests=True,
        night_prior_physics=p['night_prior_physics'],night_prior_commands=p['night_prior_commands'],night_worst_physics=p['night_prior_physics']+p['worst_physics'],night_worst_commands=p['night_prior_commands']+44,
        main_seconds=600,audit_seconds=300,backup_seconds=600,source_sync_seconds=180,all_seconds=1800,optimizer=0,eight_started=False))
    print('T471_REGISTERED',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+p['all_seconds']<p['absolute_all_stop'] and now+p['main_seconds']<p['absolute_native_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T471','--why=Matched fixed resetteacher recovery diagnostic',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T471_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
