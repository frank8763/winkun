"""One bounded fresh-policy RL pilot, no expired-night continuation."""
from t439_core import *
import subprocess,ast

def prepare():
    assert not OUT.exists()
    assert read(ROOT/'stage/runs/t436_frozen_failure_trace/audit.json')['state']=='PASS'
    control=ROOT/'stage/runs/t437_joint_closed_loop_rl'
    assert read(ROOT/'stage/runs/t438_t437_audit_recovery/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T437.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(control/'protocol.json'))
    panel=set(p['panel_ids']);focus=list(p['focus_order'])
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    cap=sum(jobs[i]['case']['step_cap'] for i in panel)
    worst=8*sum(jobs[focus[k%7]]['case']['step_cap'] for k in range(4))+4*cap+sum(j['case']['step_cap'] for j in jobs.values())
    commands=4+32+4*len(panel)+329
    assert worst<=13000000 and commands+8<=550
    now=time.time();hard=read(control/'protocol.json')['deadline_all']
    assert now+3600+900+900<hard
    p.update(id='T439_MATCHED_T437_REPRESENTATION_RL_CONTROL',out=str(OUT),
        relay_stage='representation_rl',
        stage_budgets=dict(representation_rl=dict(main_seconds=3600,audit_seconds=900,max_physics=13000000,max_commands=542,max_bytes=19*2**20)),
        root_limits=dict(max_physics=13000000,max_commands=550,max_bytes=21*2**20),
        rounds=dict(representation_rl=4),maximum_audit_Adam=dict(representation_rl=64),
        deadline_learning=min(now+2400,hard-2700),deadline_native=min(now+4200,hard-1500),deadline_all=hard,
        validation_reserve_seconds=1200,backup_seconds=600,source_seconds=300,
        preflight_discarded_Adam=6,preflight_zero_physics_commands=8,
        new_authorization='Nov9 user asks why negative transfer and explicitly continue; new bounded trainability control, no T437 restart or extension.',
        pilot='Same starting serialized T427 research model, same freshAdam/lr/reward/mean auxiliaries/62+211+7/focus_order/panel/seeds as T437 first4; only existing encoder/core additionally trainable (10722 vs4450). No new module/capacity/teacher/route.',
        trainability_is_only_intended_factor=True,mean_retention_for_both_profiles=True,
        controlled_horizon_rounds=4,control_first4_then_final_kept_same_as_T437_full8=True,
        source_control_protocol=str(control/'protocol.json'),source_control_summary=str(control/'fresh_rl/summary.json'),
        source_control_recovery_audit=str(ROOT/'stage/runs/t438_t437_audit_recovery/audit.json'),
        new_RL_steps_max=32,shadow_Adam_max=32,maximum_new_training_episodes=16,maximum_exact_training_replays=16,
        historical_overlap_must_be_reported=True,full_stable_six_gate_NOT_IN_THIS_PROTOCOL=True,
        shared_completed_plus_this_max_physics=12246664+13000000)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{
        str(DOC/'T439_PLAN.md'),str(ROOT/'stage/runs/t436_frozen_failure_trace/audit.json'),
        str(ROOT/'stage/runs/t432_offline_night/data/summary.json'),str(ROOT/'stage/runs/t432_offline_night/data/audit.json')}
    for x in read(p['relay_dataset_path'])['result']['anchors_c']:paths.add(x['path'])
    p['inputs']={x:sha(x) for x in sorted(paths)}
    paths={str(control/'protocol.json'),str(control/'fresh_rl/summary.json'),str(control/'fresh_rl/development.json'),str(control/'fresh_rl/cases.jsonl'),str(ROOT/'stage/runs/t438_t437_audit_recovery/audit.json')}
    p['inputs'].update({x:sha(x) for x in paths})
    before=E.N.K.S.used(ROOT/'stage/runs');assert before+p['root_limits']['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p)
    save(DOC/'T439_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=before,
        global_full_reserved=before+p['root_limits']['max_bytes'],root=p['root_limits'],
        worst_physics=worst,worst_commands=commands+8,panel_count=len(panel),
        native_training=16,native_exact_repeats=16,maximum_live_Adam=32,maximum_shadow_Adam=32,
        independent_discarded_historical_Adam=64,preflight_discarded_Adam=6,
        deadlines={k:p['deadline_'+k] for k in ['learning','native','all']},
        gate_capacity_not_reserved=True,no_stable_six_claim_without_separate_full_gate=True))
    print('T439_REGISTERED',len(panel),worst,flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    spec=importlib.util.spec_from_file_location('stage439_preflight_producer',HERE/'run.py')
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    assert producer.stage_source(p,'representation_rl')[0]['policy_hash']==p['expected_policy_hash']
    assert producer.__file__==str(HERE/'run.py')
    for label in ['A','B']:
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'probe',str(OUT/'protocol.json'),'representation_rl',label,str(OUT/f'clone_{label}.json')],cwd=ROOT,check=True,timeout=300)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'contract.py'),'compare',str(OUT/'protocol.json'),'representation_rl',str(OUT/'clone_A.json'),str(OUT/'clone_B.json'),str(OUT/'clone_comparison.json')],cwd=ROOT,check=True,timeout=60)
    ban_learning();counts=mp.get_context('spawn').RawArray('q',5);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),relay_stage='representation_rl',native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[x.pid for x in pool.children]
        requests=[dict(model=p['student_start_model'],job=p['main_jobs'][0],mode='student',seed=None,cancel_probe=True)]*4
        assert all(x['state']=='CANCEL_PASS' for x in pool.request(requests)) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=6,zero_physics_commands=8,
        independent_fresh_and_mature_gradients_Actor_Adam_exact=True,rollback_exact=True,
        explicit_pinned_producer_import_contract=True,
        actual_four_workers_cancelled_zero_native_and_stopped=True,
        synthetic_engineering_not_navigation=True,no_models_retained=True))
    print('T439_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    now=time.time();assert now+3600+900+900<p['deadline_all'];limit=p['deadline_all']-now-5
    with (DOC/'T439_CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T439','--why=Bounded Stage fresh-policy RL pilot','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,
        protocol_sha256=sha(OUT/'protocol.json'),offline_science=True,not_a_restart=True))
    print('T439_ACTUALLY_LAUNCHED',child.pid,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
