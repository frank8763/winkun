"""Register and launch the new offline daytime protocol exactly once."""
from t425_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types
def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True);old=read(ROOT/'stage/runs/t422_feedback_recovery_night/protocol.json')
    keys=['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','left5_jobs','right_job','left_history_files','right_history_file','main_jobs','fresh_jobs','reload_jobs','centres','deployment','initial_model','full_bytes','fresh_ledger','inputs','phase','query_sources','teacher_model','teacher_policy_hash']
    p={k:copy.deepcopy(old[k]) for k in keys};prev=read(ROOT/'stage/runs/t424_frozen_six_acceptance/protocol.json');findings=read(ROOT/'stage/docs/acceptance_20261108/FINDINGS.json');assert findings['six_stable'] is False
    rows=[json.loads(line) for line in (ROOT/'stage/runs/t424_frozen_six_acceptance/cases.jsonl').read_text().splitlines()];development=[r for r in rows if r['role'] in ['wide54','old275']];assert len(development)==329
    anchors_=[];seen=set()
    for row in rows:
        if not row['full'] or not row['success'] or row['array_digest'] in seen:continue
        seen.add(row['array_digest']);anchors_.append(dict(path=row['path'],sha256=row['sha256'],row=row))
    assert 0<len(anchors_)<=80
    jobs={j['id']:j for j in [*p['main_jobs'],*prev['old275_jobs']]};targets=[];mapping={'wide117_80_dev6_LEFT_02':'0015','wide117_80_dev6_LEFT_06':'0021','wide117_80_dev6_LEFT_15':'0030','wide117_80_dev6_LEFT_19':'0032'}
    for failure in findings['failures']:
        identity=failure['id'];entry=dict(job=jobs[identity],kind='safe_reset' if failure['role']=='old275' else 'feedback')
        if entry['kind']=='feedback':
            path=ROOT/'stage/runs/t424_frozen_six_acceptance/episodes'/(mapping[identity]+'.npz');a=E.N.S.arrays(path);assert a['end_flags'][-1,1];entry.update(reference_path=str(path),reference_sha256=sha(path),same_initial_state=False)
        targets.append(entry)
    assert len(targets)==19
    reference_gain=read(ROOT/'stage/runs/t422_feedback_recovery_night/search/summary.json')['result']['best']['gains'];g2=list(reference_gain);g2[0]*=1.5;g2[1]*=1.25;g3=list(reference_gain);g3[0]*=.6;g3[1]*=.75;g4=list(reference_gain);g4[2]*=.5;g4[5]=.5
    zone=ZoneInfo('Asia/Shanghai');epochs={name:datetime(2026,11,8,hour,0,tzinfo=zone).timestamp() for name,hour in [('learning',15),('native',16),('all',17)]}
    stage_budgets=dict(data=dict(main_seconds=1800,audit_seconds=600,max_physics=3000000,max_commands=200,max_bytes=16*2**20),arm_a=dict(main_seconds=3600,audit_seconds=1800,max_physics=16000000,max_commands=700,max_bytes=32*2**20),arm_b=dict(main_seconds=3600,audit_seconds=900,max_physics=16000000,max_commands=700,max_bytes=32*2**20),gate=dict(main_seconds=2400,audit_seconds=600,max_physics=6000000,max_commands=200,max_bytes=39*2**20),eight=dict(main_seconds=900,audit_seconds=300,max_physics=2300000,max_commands=100,max_bytes=4*2**20))
    old8=[json.loads(l)['job'] for l in (ROOT/'stage/runs/t255_overnight/8m.jsonl').read_text().splitlines()];assert len(old8)==2;eight_jobs=[]
    for template in old8:
        side=template['case']['side'];base=dict(template,id='wide8_'+side+'_centre',wide_course=True,circle_shift_x_m=.6,case=dict(template['case'],step_cap=40000),delta_x_m=0.,delta_y_m=0.,delta_yaw_deg=0.)
        for key in ['exact_snapshot','near_snapshot']:base.pop(key,None)
        eight_jobs.append(base)
        k=0
        for dx in [-.03,0,.03]:
            for dy in [-.03,0,.03]:
                for yaw in [-2.,0.,2.]:eight_jobs.append(dict(base,id=f'wide8_{side}_dev_{k:02}',delta_x_m=dx,delta_y_m=dy,delta_yaw_deg=yaw));k+=1
    assert len(eight_jobs)==56
    p.update(id='T425_BOUNDED_OFFLINE_RECOVERY_RETENTION_DIRECT8',out=str(OUT),student_start_model=prev['student_start_model'],student_start_sha256=prev['student_start_sha256'],expected_policy_hash=prev['expected_policy_hash'],anchors=anchors_,targets=targets,old275_jobs=prev['old275_jobs'],initial_success_ids=sorted(r['job_id'] for r in development if r['success']),initial_development_rows=development,wide_failed_jobs=[x['job'] for x in targets if x['kind']=='feedback'],feedback_gains=[reference_gain,g2,g3,g4],teacher_first=17,seed=20261108425,epochs=8,old_weight=1.,broad_old_weight=1.,left_weight=.25,right_weight=.25,query_weight=1.,stage_budgets=stage_budgets,root_limits=dict(max_physics=44000000,max_commands=2000,max_bytes=128*2**20),deadline_learning=epochs['learning'],deadline_native=epochs['native'],deadline_all=epochs['all'],backup_seconds=900,source_seconds=180,preflight_zero_physics_commands=64,preflight_fresh_clone_Adam=2,a_steps=120,full_cycle_reserve_seconds=dict(a=2400,b=1400),update_reserve_seconds=dict(a=1400,b=300),validation_reserve_seconds=1200,maximum_rounds=dict(a=1,b=2),feedback_speed_contract="Current and reference planar measured speed are float32 state56 in m/s; historical .5*state113 was LF steering encoder, corrected only in T425; not an isolated gain-factor ablation.",eight_jobs=eight_jobs,eight_horizon='New8m factory with same80s as current6m, same0.2mGOAL/hold100/plant;56 frozen diagnostics only,no8mtraining or stability claim.',new_authorization='UserNov8 offline local background to17:00, multiple finite mechanisms; six full gates before8, seven explicitly skipped.',old_training_audit='FAIL_RECONSTRUCTION_CAUSE_UNRESOLVED',query_full_bytes=p['full_bytes'])
    p['fresh_ledger']=prev['fresh_ledger'].copy()
    for root in [ROOT/'stage/runs/t424_frozen_six_acceptance']:
        for f in root.rglob('*.jsonl'):
            if f.name in ['cases.jsonl','replays.jsonl']:p['fresh_ledger'][str(f)]=sha(f)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{x['path'] for x in anchors_}|{x['reference_path'] for x in targets if x['kind']=='feedback'}|{p['student_start_model'],str(ROOT/'stage/docs/acceptance_20261108/FINDINGS.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T424.json'),str(DOC/'T425_PROVENANCE.json'),str(DOC/'PLAN.md')}
    p['inputs']={f:sha(f) for f in sorted(paths)};used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()+1800+600+3600+1800<epochs['all'] and time.time()+1800+600+3600<epochs['learning']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=used,global_full_reserved=used+p['root_limits']['max_bytes'],new_limits=p['root_limits'],stages=stage_budgets,prior_today_physics=5606702,whole_day_physics_worst=49606702,preflight_zero_commands=64,new_clone_Adam=2,prior_historical_Adam_probes=5,learning_absolute='2026-11-08 15:00',native_absolute='2026-11-08 16:00',all_absolute='2026-11-08 17:00',six_before_eight=True,seven_skipped=True,no_scientific_network_dependency=True));print('T425_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for label in ['A','B']:
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'new_contract.py'),'probe',label],cwd=ROOT,check=True,timeout=300)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'new_contract.py'),'compare'],cwd=ROOT,check=True,timeout=60)
    ban_learning()
    for name in ['t425_core.py','t425_learning.py','new_contract.py','run.py','audit.py','driver.py','runner.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage425_preflight_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    folder=OUT/'preflight_native';folder.mkdir();w=None;geometry=[]
    with (folder/'console.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=E.N.S.P.C.B.setup(folder/'geometry',E.N.science(p));original=copy.deepcopy(module.ACTIVE)
        for job in p['eight_jobs']:
            module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=.6;env,obs,actual=E.N.K.R.task_env(create,job)
            try:
                obs=E.N.K.C.perturb_anchor(env,obs,job);pose=np.asarray(env._pose()[:3]);distance=float(np.linalg.norm(env.goal_xy-pose[:2]));geometry.append(dict(id=job['id'],valid=not env._collision(),actual_factory=actual,distance_m=distance,goal=env.goal_xy.tolist(),pose=pose.tolist(),initial_integration_sha256=digest(E.N.K.R.integration(env)),scene_sha256=E.N.K.C.model_hash(env),native=0));assert 7.8<distance<8.2
            finally:w.events.pop(id(env),None);env.close()
        w.close();w=None
    save(OUT/'eight_geometry.json',geometry);assert all(g['valid'] for g in geometry if g['id'].endswith('centre'))
    counts=mp.get_context('spawn').RawArray('q',5);f=folder/'pool';f.mkdir();pool=Pool(dict(p,out=str(f),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['wide_failed_jobs'][0],mode='student',seed=None,cancel_probe=True)]*4);assert all(x['state']=='CANCEL_PASS' for x in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,new_discarded_clone_Adam=2,zero_physics_commands=64,cross_process_optimizer_contract=read(OUT/'preflight_clone_comparison.json'),actual_eight_geometry=56,actual_four_workers_zero_step_cancel_and_exit=True,recursive_entry_globals=True,old_training_audit_NOT_repaired=True));E.N.manifest(OUT);print('T425_PREFLIGHT_PASS',flush=True)
def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+1800+600+3600<p['deadline_learning'];limit=p['deadline_all']-now-10;assert limit>7200
    with (DOC/'DAY_CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T425','--why=Offline bounded recovery retention direct8 conditional research','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,six_before_eight=True,seven_skipped=True));print('T425_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
