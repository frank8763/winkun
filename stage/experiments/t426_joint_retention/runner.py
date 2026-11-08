"""Register/preflight/launch the new bounded joint retention pilot once."""
from t426_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types
import t426_contract as CONTRACT

def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True);previous=read(ROOT/'stage/runs/t425_offline_day_chain/protocol.json');record=read(ROOT/'stage/runs/t425_offline_day_chain/a/blocks/0/result.json');check=record['check'];source=record['proposed_model'];assert check['wide54']['success']==54 and check['old275']['success']==264
    keys=['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','main_jobs','fresh_jobs','reload_jobs','centres','deployment','initial_model','full_bytes','fresh_ledger','inputs','phase','old275_jobs','eight_jobs']
    p={k:copy.deepcopy(previous[k]) for k in keys};proofs={r['job_id']:r for r in check['rows']};old_ids={j['id'] for j in p['old8_jobs']};anchor_jobs=[*p['main_jobs'],*[j for j in p['old8_jobs'] if proofs[j['id']]['success']]];assert len(anchor_jobs)==61
    failures=[r for r in check['rows'][54:] if not r['success']];assert len(failures)==11;jobmap={j['id']:j for j in p['old275_jobs']};recoveries=[jobmap[r['job_id']] for r in failures]
    zone=ZoneInfo('Asia/Shanghai');cut={name:datetime(2026,11,8,h,m,tzinfo=zone).timestamp() for name,h,m in [('learning',20,0),('native',20,30),('all',21,0)]}
    stages=dict(data=dict(main_seconds=1200,audit_seconds=300,max_physics=3100000,max_commands=110,max_bytes=32*2**20),learn=dict(main_seconds=3600,audit_seconds=1500,max_physics=14000000,max_commands=670,max_bytes=16*2**20),gate=dict(main_seconds=1800,audit_seconds=600,max_physics=5900000,max_commands=170,max_bytes=36*2**20),eight=dict(main_seconds=900,audit_seconds=300,max_physics=2300000,max_commands=65,max_bytes=4*2**20))
    p.update(id='T426_CURRENT_A_FAMILY_BALANCED_RETENTION_DIRECT8',out=str(OUT),student_start_model=source['path'],student_start_sha256=source['sha256'],expected_policy_hash=source['policy_hash'],source_model=source,anchor_jobs=anchor_jobs,anchor_proofs=[proofs[j['id']] for j in anchor_jobs],recovery_jobs=recoveries,initial_success_ids=sorted(check['success_ids']),allowed_teacher_first=[1,0],maximum_steps=120,checkpoints=[60,120],stage_budgets=stages,root_limits=dict(max_physics=25400000,max_commands=1100,max_bytes=96*2**20),deadline_learning=cut['learning'],deadline_native=cut['native'],deadline_all=cut['all'],validation_reserve_seconds=1200,update_window_seconds=600,cycle_window_seconds=1600,backup_seconds=600,source_seconds=180,preflight_zero_physics_commands=64,preflight_fresh_clone_Adam=2,old_training_audit='T422_STILL_FAIL_NOT_REPAIRED',new_authorization='User2026-11-08 afterT425final asks urgently repair andacceptsixthen8,skip7. Newboundedgoal, never extend/restart endedT425.',research_trajectory='SamefreshAdam120 steps, checkpoint60then120; a rejected60 is not deployed but fixed research trajectory may continue.60all329safestops120. Finalonlyinitial318subsetplusnewrepair highestcount, tieearlier, exactfullActorAdamreadback.',pilot_factors='NewA54initializer/current54+7fullsuccessfulhistory/familybalancedretention/11safeactualrecovery/noobsoleteT406aux together; engineeringpilot, noisolatedcausalclaimororiginalalgorithm.')
    assert len(p['initial_success_ids'])==318 and len(set(p['initial_success_ids']))==318
    freshids={j['id'] for j in p['fresh_jobs']}
    for dirname in ['stage/runs/t424_frozen_six_acceptance','stage/runs/t425_offline_day_chain']:
        for path in (ROOT/dirname).rglob('*.jsonl'):
            if path.name not in ['cases.jsonl','replays.jsonl']:continue
            for line in path.read_text().splitlines():assert json.loads(line).get('job_id') not in freshids
            p['fresh_ledger'][str(path)]=sha(path)
    fixtures,labels=CONTRACT.fixture_sources(p)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{x['path'] for x in [*fixtures,*labels]}|{source['path'],str(ROOT/'stage/runs/t425_offline_day_chain/a/blocks/0/result.json'),str(ROOT/'stage/runs/t425_offline_day_chain/protocol.json'),str(ROOT/'stage/runs/t425_offline_day_chain/data/targets.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T425.json'),str(DOC/'PLAN.md')}
    p['inputs']={path:sha(path) for path in sorted(paths)};used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    exact_data=sum(j['case']['step_cap'] for j in anchor_jobs)+3*sum(j['case']['step_cap'] for j in recoveries);exact_learn=2*sum(j['case']['step_cap'] for j in [*p['main_jobs'],*p['old275_jobs']]);exact_gate=sum(j['case']['step_cap'] for j in p['reload_jobs'])*2;exact_eight=sum(j['case']['step_cap'] for j in p['eight_jobs']);assert exact_data<=stages['data']['max_physics'] and exact_learn<=stages['learn']['max_physics'] and exact_gate<=stages['gate']['max_physics'] and exact_eight<=stages['eight']['max_physics'];assert 23664298+sum([exact_data,exact_learn,exact_gate,exact_eight])<50000000
    assert sum(x['max_bytes'] for x in stages.values())+8*2**20<=p['root_limits']['max_bytes']
    assert time.time()+1200+300+3600+1500<cut['all'] and time.time()+1200+300+3600<cut['learning']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=used,full_global_reserved=used+p['root_limits']['max_bytes'],root=p['root_limits'],stages=stages,exact_physics=dict(data=exact_data,learn=exact_learn,gate=exact_gate,eight=exact_eight),prior_today_physics=23664298,whole_day_exact_max=23664298+exact_data+exact_learn+exact_gate+exact_eight,newfullprefAdam=2,newzero_physics_commands=64,deadline=cut,old17cutoff_NOT_extended_new_goal=True));print('T426_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for label in ['A','B']:subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t426_contract.py'),'probe',label],cwd=ROOT,check=True,timeout=180)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t426_contract.py'),'compare'],cwd=ROOT,check=True,timeout=60)
    ban_learning()
    for name in ['t426_core.py','t426_learning.py','t426_contract.py','run.py','audit.py','driver.py','runner.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage426_check_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(name,check(fun.__code__))
    folder=OUT/'preflight_native';folder.mkdir();geometry=[]
    with (folder/'console.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=E.N.S.P.C.B.setup(folder/'geometry',E.N.science(p));original=copy.deepcopy(module.ACTIVE)
        try:
            for job in p['eight_jobs']:
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=.6;env,obs,actual=E.N.K.R.task_env(create,job)
                try:
                    obs=E.N.K.C.perturb_anchor(env,obs,job);pose=np.asarray(env._pose()[:3]);distance=float(np.linalg.norm(env.goal_xy-pose[:2]));assert 7.8<distance<8.2;geometry.append(dict(id=job['id'],valid=not env._collision(),distance_m=distance,actual_factory=actual,goal=env.goal_xy.tolist(),pose=pose.tolist(),scene_sha256=E.N.K.C.model_hash(env),native=0))
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'eight_geometry.json',geometry);assert all(g['valid'] for g in geometry if g['id'].endswith('centre'))
    counts=mp.get_context('spawn').RawArray('q',5);f=folder/'pool';f.mkdir();pool=Pool(dict(p,out=str(f),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['main_jobs'][0],mode='student',seed=None,cancel_probe=True)]*4);assert all(x['state']=='CANCEL_PASS' for x in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids);save(OUT/'preflight.json',dict(state='PASS',native=0,new_discarded_Adam=2,zero_physics_commands=64,independent_clone=read(OUT/'clone_comparison.json'),current_A_native_successes_not_yet_reconstructed=True,four_actual_workers_cancelled_and_exited=True,entry_globals_and_inputs=True,old_T422_audit_NOT_repaired=True));E.N.manifest(OUT);print('T426_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+1200+300+3600<p['deadline_learning'];limit=min(10800,p['deadline_all']-now-10);assert limit>7200
    with (DOC/'CONSOLE.log').open('xb') as log:child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T426','--why=Bounded six joint acceptance then direct eight diagnostics','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,six_before_eight=True,seven_skipped=True,new_goal_not_T425_restart=True));print('T426_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
