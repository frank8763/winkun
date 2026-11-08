"""Register the fixed-representation recovery pilot; do not resume old runs."""
from t427_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types
import t427_contract as CONTRACT

def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True);old=read(ROOT/'stage/runs/t426_joint_retention/protocol.json');summary=read(ROOT/'stage/runs/t426_joint_retention/learn/summary.json');assert summary['state']=='COMPLETE' and read(ROOT/'stage/runs/t426_joint_retention/learn/audit.json')['state']=='PASS';record=summary['result']['records'][-1];check=read(record['development_file']);source=record['proposed_model'];assert check['wide54']['success']==49 and check['old275']['success']==273
    keys=['source_protocol','raw_sigma','lr','epochs','ppo_clip','max_logratio','old8_jobs','main_jobs','fresh_jobs','reload_jobs','centres','deployment','initial_model','full_bytes','fresh_ledger','inputs','phase','old275_jobs','eight_jobs']
    p={k:copy.deepcopy(old[k]) for k in keys};proofs={r['job_id']:r for r in check['rows']};jobs=[*p['main_jobs'],*p['old8_jobs']];anchors=[j for j in jobs if proofs[j['id']]['success']];assert len(anchors)==57;alljobs={j['id']:j for j in [*p['main_jobs'],*p['old275_jobs']]};failed=[r for r in check['rows'] if not r['success']];recoveries=[alljobs[r['job_id']] for r in failed];assert len(recoveries)==7
    cut={name:datetime(2026,11,8,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp() for name,h,m in [('learning',20,0),('native',20,30),('all',21,0)]};stages=dict(data=dict(main_seconds=900,audit_seconds=180,max_physics=3100000,max_commands=100,max_bytes=28*2**20),learn=dict(main_seconds=1500,audit_seconds=300,max_physics=7000000,max_commands=340,max_bytes=32*2**20),gate=dict(main_seconds=1200,audit_seconds=300,max_physics=3590000,max_commands=100,max_bytes=16*2**20),eight=dict(main_seconds=900,audit_seconds=180,max_physics=2240000,max_commands=65,max_bytes=4*2**20))
    teacher=old['source_model'];p.update(id='T427_FROZEN_REPRESENTATION_66_OUTPUT_REPAIR',out=str(OUT),student_start_model=source['path'],student_start_sha256=source['sha256'],expected_policy_hash=source['policy_hash'],source_model=source,teacher_A_model=teacher['path'],teacher_A_sha256=teacher['sha256'],teacher_A_policy_hash=teacher['policy_hash'],anchor_jobs=anchors,anchor_proofs=[proofs[j['id']] for j in anchors],recovery_jobs=recoveries,initial_success_ids=sorted(check['success_ids']),allowed_teacher_first=[1,0],steps=120,stage_budgets=stages,root_limits=dict(max_physics=13540000,max_commands=1100,max_bytes=88*2**20),deadline_learning=cut['learning'],deadline_native=cut['native'],deadline_all=cut['all'],validation_reserve_seconds=900,backup_seconds=600,source_seconds=180,preflight_zero_physics_commands=64,preflight_fresh_clone_Adam=2,new_authorization='UserNov8 afterT426 ended saysreason andcontinue solving urgently. Newfrozenrepresentation/outputonlyhypothesis, notadditionalT426rounds. Existing21:00all deadline retained.',pilot='Current120initializer322success,allbutexistingextra.2W/biasfrozen66,211current-selfdistillation,newactual7recovery; jointengineeringnotoriginalnotisolatedcausal. No map/planner/waypoint/runtime teacher/router or7m.',conditional_reserve='Eachnextstage requires fullworstremainingphysics/cmd/disk/time; sumstageceilings mayexceedroot and later stages canbeskipped. Root/daycaps NEVERexceeded.',old_training_audit='T422_STILL_FAIL_NOT_REPAIRED')
    assert len(p['initial_success_ids'])==322
    for path in (ROOT/'stage/runs/t426_joint_retention').rglob('*.jsonl'):
        if path.name not in ['cases.jsonl','replays.jsonl']:continue
        for line in path.read_text().splitlines():assert json.loads(line).get('job_id') not in {j['id'] for j in p['fresh_jobs']}
        p['fresh_ledger'][str(path)]=sha(path)
    fixtures,labels=CONTRACT.fixture_sources(p)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{x['path'] for x in [*fixtures,*labels]}|{source['path'],teacher['path'],str(ROOT/'stage/runs/t426_joint_retention/learn/summary.json'),str(ROOT/'stage/runs/t426_joint_retention/learn/audit.json'),record['development_file'],str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T426.json'),str(DOC/'PLAN.md')};p['inputs']={path:sha(path) for path in sorted(paths)}
    used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30;assert sum(x['max_bytes'] for x in stages.values())+8*2**20<=p['root_limits']['max_bytes'];assert time.time()+900+180+1500<cut['learning']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',global_before=used,full_global_reserved=used+p['root_limits']['max_bytes'],root=p['root_limits'],stages=stages,prior_today_physics=36459930,whole_day_hard_max=49999930,remaining_today=13540070,actual_stage_admission_mandatory=True,all21stop=True,seven_skipped=True));print('T427_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for label in ['A','B']:subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t427_contract.py'),'probe',label],cwd=ROOT,check=True,timeout=180)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t427_contract.py'),'compare'],cwd=ROOT,check=True,timeout=60)
    ban_learning()
    for name in ['t427_core.py','t427_learning.py','t427_contract.py','run.py','audit.py','driver.py','runner.py','transport.py']:
        spec=importlib.util.spec_from_file_location('stage427_check_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
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
        ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['main_jobs'][0],mode='actor_teacher',seed=None,cancel_probe=True,teacher_model=p['teacher_A_model'],teacher_sha256=p['teacher_A_sha256'])]*4);assert all(x['state']=='CANCEL_PASS' for x in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids);save(OUT/'preflight.json',dict(state='PASS',native=0,new_discarded_Adam=2,zero_physics_commands=64,independent_clone=read(OUT/'clone_comparison.json'),current_native_successes_not_yet_reconstructed=True,four_actual_workers_cancelled_and_exited=True,entry_globals_and_inputs=True,old_T422_audit_NOT_repaired=True));E.N.manifest(OUT);print('T427_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+900+180+600<p['deadline_learning'];limit=min(10800,p['deadline_all']-now-10);assert limit>3600
    with (DOC/'CONSOLE.log').open('xb') as log:child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T427','--why=Bounded six joint acceptance then direct eight diagnostics','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,six_before_eight=True,seven_skipped=True,new_goal_not_T426_restart=True));print('T427_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
