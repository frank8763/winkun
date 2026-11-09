"""One bounded fresh-policy RL pilot, no expired-night continuation."""
from t437_core import *
import subprocess,ast

def prepare():
    assert not OUT.exists()
    assert read(ROOT/'stage/runs/t436_frozen_failure_trace/audit.json')['state']=='PASS'
    p=copy.deepcopy(read(ROOT/'stage/runs/t432_offline_night/protocol.json'))
    focus=list(p['recovery_target_ids']);assert len(focus)==7
    source=[json.loads(x) for x in Path(p['current_cases_path']).read_text().splitlines()];byid={x['job_id']:x for x in source}
    panel=set(focus)|{j['id'] for j in p['old8_jobs']}|{j['id'] for j in p['additional_old_jobs']}
    panel.update(['wide117_80_dev6_LEFT_00','wide117_80_dev6_LEFT_07','dev_task6_09','dev_task6_24',
        'wide117_80_dev6_LEFT_15','wide117_80_dev6_LEFT_18','wide117_80_dev6_LEFT_21','dev_task6_13','dev_task6_15'])
    for prefix in ['wide117_80_dev6_LEFT_','wide117_80_dev6_RIGHT_']:
        panel.update([x['job_id'] for x in source if x['job_id'].startswith(prefix) and x['success']][:4])
    assert len(panel)<=32 and panel<=set(byid)
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    cap=sum(jobs[i]['case']['step_cap'] for i in panel)
    training=8*sum(jobs[focus[k%7]]['case']['step_cap'] for k in range(8))
    dev=sum(j['case']['step_cap'] for j in jobs.values())
    worst=training+8*cap+dev
    commands=4+64+8*len(panel)+329
    assert worst<22000000 and commands<=750
    now=time.time()
    p.update(id='T437_JOINT_RECOVERY_AUXILIARY_FRESH_POLICY_RL',out=str(OUT),
        relay_dataset_path=str(ROOT/'stage/runs/t432_offline_night/data/summary.json'),
        anchor_source_model_sha256=p['student_start_sha256'],
        stage_budgets=dict(fresh_rl=dict(main_seconds=5400,audit_seconds=1800,max_physics=22000000,max_commands=750,max_bytes=28*2**20)),
        root_limits=dict(max_physics=22000000,max_commands=758,max_bytes=30*2**20),
        rounds=dict(fresh_rl=8),maximum_audit_Adam=dict(fresh_rl=128),
        focus_order=focus,panel_ids=sorted(panel),seed_base=20261109437,
        deadline_learning=now+4800,deadline_native=now+6600,deadline_all=now+9900,
        validation_reserve_seconds=1200,backup_seconds=600,source_seconds=300,
        preflight_discarded_Adam=6,preflight_zero_physics_commands=8,
        new_authorization='Nov9 explicit entrustment and continue: independent bounded daytime pilot, not extension of expired overnight.',
        pilot='Fixed62 actual starting-policy successes+211 historical self targets+7 verified recoveries; 8 registered fresh four-peer on-policy RL rounds, existing4450 trainable; same reward and iid density. No new teacher, representation or deployed module.',
        source_Adam_not_inherited=True,new_RL_steps_max=64,shadow_Adam_max=64,
        no_early_two_rejection_termination=True,maximum_new_training_episodes=32,maximum_exact_training_replays=32,
        full329_only_after_research_update_kept=True,fresh16=None,reload78=None,eight_started=False,
        full_stable_six_gate_NOT_IN_THIS_PROTOCOL=True,
        metadata_root_budget=2*2**20,
        old_T422_history_FAIL_remains=True,
        scientific_goal='Does new true on-policy RL with existing recovery/retention auxiliaries acquire any compatible capability? Engineering baseline, not claimed new algorithm or isolated causal comparison with supervised T432.')
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{
        str(DOC/'T437_PLAN.md'),str(ROOT/'stage/runs/t436_frozen_failure_trace/audit.json'),
        str(ROOT/'stage/runs/t432_offline_night/data/summary.json'),str(ROOT/'stage/runs/t432_offline_night/data/audit.json')}
    for x in read(p['relay_dataset_path'])['result']['anchors_c']:paths.add(x['path'])
    p['inputs']={x:sha(x) for x in sorted(paths)}
    before=E.N.K.S.used(ROOT/'stage/runs');assert before+p['root_limits']['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p)
    save(DOC/'T437_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=before,
        global_full_reserved=before+p['root_limits']['max_bytes'],root=p['root_limits'],
        worst_physics=worst,worst_commands=commands+8,panel_count=len(panel),
        native_training=32,native_exact_repeats=32,maximum_live_Adam=64,maximum_shadow_Adam=64,
        independent_discarded_historical_Adam=128,preflight_discarded_Adam=6,
        deadlines={k:p['deadline_'+k] for k in ['learning','native','all']},
        gate_capacity_not_reserved=True,no_stable_six_claim_without_separate_full_gate=True))
    print('T437_REGISTERED',len(panel),worst,flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    for label in ['A','B']:
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'probe',str(OUT/'protocol.json'),'fresh_rl',label,str(OUT/f'clone_{label}.json')],cwd=ROOT,check=True,timeout=300)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'compare',str(OUT/'protocol.json'),'fresh_rl',str(OUT/'clone_A.json'),str(OUT/'clone_B.json'),str(OUT/'clone_comparison.json')],cwd=ROOT,check=True,timeout=60)
    ban_learning();counts=mp.get_context('spawn').RawArray('q',5);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),relay_stage='fresh_rl',native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[x.pid for x in pool.children]
        requests=[dict(model=p['student_start_model'],job=p['main_jobs'][0],mode='student',seed=None,cancel_probe=True)]*4
        assert all(x['state']=='CANCEL_PASS' for x in pool.request(requests)) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=6,zero_physics_commands=8,
        independent_fresh_and_mature_gradients_Actor_Adam_exact=True,rollback_exact=True,
        actual_four_workers_cancelled_zero_native_and_stopped=True,
        synthetic_engineering_not_navigation=True,no_models_retained=True))
    print('T437_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    now=time.time();assert now+5400+1800+900<p['deadline_all'];limit=p['deadline_all']-now-5
    with (OUT/'driver_console.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T437','--why=Bounded Stage fresh-policy RL pilot','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,
        protocol_sha256=sha(OUT/'protocol.json'),offline_science=True,not_a_restart=True))
    print('T437_ACTUALLY_LAUNCHED',child.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
