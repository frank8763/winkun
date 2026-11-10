"""Independent frozen counterexample diagnosis after the closed T468 relay."""
from pathlib import Path
import copy,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t469_counterexample_recovery';DOC=ROOT/'stage/docs/overnight_20261110'
sys.path.insert(0,str(HERE))
import run as T
read,save,sha=T.read,T.save,T.sha
IDS=['dev_task6_03','dev_task6_15','wide117_80_dev6_LEFT_15','wide117_80_dev6_LEFT_17']

def prepare():
    assert not OUT.exists(),'Neverrestart'
    prior=ROOT/'stage/runs/t468_offline_recovery_night';old=read(prior/'protocol.json');report=read(DOC/'FINAL_REPORT.json')
    assert report['state']=='COMPLETE_AUDITED_LOCAL' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert report['completed']==['occupancy_0','duration_0'] and not report['six_gate']
    for f in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(prior/f)['pid'])).exists()
    assert all(not Path('/proc',str(v)).exists() for v in read(prior/'SEALED.json')['scientific_pids'])
    initial=old['initial_candidate'];failed=read(prior/'occupancy_0/summary.json')['candidate']
    assert set(initial['success_ids'])-set(failed['success_ids'])==set(IDS)
    arows={r['job_id']:r for r in map(json.loads,(prior/'occupancy_0/cases.jsonl').read_text().splitlines()) if r['role']=='FINAL329'}
    history=ROOT/'stage/runs/t463_evening_2000/ppo/cases.jsonl';brows={r['job_id']:r for r in map(json.loads,history.read_text().splitlines()) if r['role']=='FINAL329'}
    p=copy.deepcopy(old['science']);jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    p.update(id='T469_FROZEN_COUNTEREXAMPLE_RECOVERY_DIAGNOSIS',out=str(OUT),learning_initial=failed['model'],teacher_model=initial['model'],protected_files=old['protected_files'],
        diagnostic_jobs=[jobs[k] for k in IDS],diagnostic_reference_student={k:arows[k] for k in IDS},diagnostic_reference_teacher={k:brows[k] for k in IDS},
        max_bytes=18*2**20,max_physics=1000000,max_commands=80,preflight_commands=24,actual_worst_commands=64,
        full_bytes=448*1024,compact_bytes=4096,model_bytes=0,main_seconds=900,learning_seconds=1,audit_seconds=300,backup_seconds=600,all_seconds=1800,
        onset_min_decision=20,onset_consecutive=3,onset_abs_steering_error=.02,diagnosisrule=dict(min_decision=20,consecutive=3,abs_steering_error_threshold=.02,nohit_fallback_decision=20),
        absolute_learning_stop=old['absolute_learning_stop'],absolute_native_stop=old['absolute_native_stop'],absolute_all_stop=old['absolute_all_stop'],
        night_prior_physics=report['physics'],night_prior_commands=report['commands'],night_prior_main_seconds=report['main_elapsed_s'],
        night_total_max_physics=old['max_physics'],night_total_max_commands=old['max_commands'],night_total_max_main_seconds=old['max_main_seconds'],
        teacher_status='Original327 is only a candidate recovery teacher; four autonomous GOALs are not evidence of recovery from rejectedA histories.',
        rejected_model_scope='After T468 is ended, user explicitly authorized next counterexample experiment. A325 is frozen diagnostic source only, never promoted or used as new student initialization. Its histories and verified recovery data may become explicitly offpolicy finite preservation support in separately registered T470; this is data reuse, not a claim that rejected data are absent.',
        user_authorization='Nov10 user: start next experiment after inspecting T468 failure report. New zero-optimizer counterexample diagnosis, not restarting or extending either closed T468 mechanism.',
        next_learning='T470 remains unregistered/unlaunched. Only all4new recovery safe+exact and independentaudit+fullbackup allow formal finite joint repair from original327 with these offpolicy counterexamples. Any recovery failure means0BC/no alternate teacher/time/threshold/seed grid.',
        frozen_original_52866_and_residual_1074=True,Actor_Adam=0,Value_Adam=0,new_optimizer=0,fresh16=None,reload78=None,six_gate=False,eight_started=False)
    p['worst_physics']=6*sum(j['case']['step_cap'] for j in p['diagnostic_jobs']);assert p['worst_physics']<=p['max_physics']
    assert p['night_prior_physics']+p['worst_physics']<=p['night_total_max_physics'] and p['night_prior_commands']+64<=p['night_total_max_commands']
    assert p['night_prior_main_seconds']+p['main_seconds']<=p['night_total_max_main_seconds']
    assert time.time()+p['all_seconds']<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop']
    assert T.M.used()+p['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['protected_files'])|{str(DOC/'T469_PLAN.md'),str(DOC/'FINAL_REPORT.json'),str(prior/'protocol.json'),str(prior/'SEALED.json'),str(prior/'occupancy_0/summary.json'),str(prior/'occupancy_0/audit.json'),str(prior/'occupancy_0/cases.jsonl'),str(history),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T468.json')}
    files|={p['learning_initial']['path'],p['teacher_model']['path']}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);T.verify(p,True)
    save(DOC/'T469_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max_bytes=p['max_bytes'],full_global=T.M.used()-T.M.used(OUT)+p['max_bytes'],max_physics=p['max_physics'],worst_physics=p['worst_physics'],commands64=True,guard80_NOT_extra_requests=True,
        night_prior_physics=p['night_prior_physics'],night_prior_commands=p['night_prior_commands'],night_worst_physics=p['night_prior_physics']+p['worst_physics'],night_worst_commands=p['night_prior_commands']+64,
        night_max_physics=p['night_total_max_physics'],night_max_commands=p['night_total_max_commands'],main_seconds=900,audit_seconds=300,all_seconds=1800,optimizer=0,eight_started=False))
    print('T469_REGISTERED',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+p['all_seconds']<p['absolute_all_stop'] and now+p['main_seconds']<p['absolute_native_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T469','--why=Bounded frozen counterexample recovery diagnosis',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')))
    print('T469_ACTUALLY_LAUNCHED',c.pid,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_control_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
