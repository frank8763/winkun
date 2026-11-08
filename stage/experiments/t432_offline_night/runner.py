"""New offline Stage night registration, numerical preflight and one launch."""
from t432_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types


def prepare():
    assert not OUT.exists();DOC.mkdir(exist_ok=True)
    prior=ROOT/'stage/runs/t431_actual_recovery'
    assert read(ROOT/'stage/docs/window_20261108_2330/DRIVER_STATUS.json')['all_scientific_processes_stopped']
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T431.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert read(ROOT/'stage/runs/t433_cold_metadata/audit.json')['state']=='PASS'
    p=copy.deepcopy(read(prior/'protocol.json'))
    original=ROOT/'stage/runs/t427_fixed_representation/learn'
    original_rows=[json.loads(line) for line in (original/'cases.jsonl').read_text().splitlines()]
    proofs={r['job_id']:r for r in original_rows}
    jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    ids=['dev_task6_08','dev_task6_09','dev_task6_10']
    assert all(proofs[i]['success'] and not proofs[i]['collision'] for i in ids)
    assert read(prior/'data/audit.json')['state']=='PASS'
    targets=read(prior/'data/targets.json');assert len(targets)==7
    cuts={k:datetime(2026,11,9,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
          for k,h,m in [('learning',7,0),('native',7,45),('all',8,30)]}
    stages=dict(data=dict(main_seconds=600,audit_seconds=180,max_physics=150000,max_commands=12,max_bytes=2*2**20),
        b=dict(main_seconds=3600,audit_seconds=900,max_physics=7000000,max_commands=340,max_bytes=5*2**20),
        c=dict(main_seconds=3600,audit_seconds=900,max_physics=7000000,max_commands=340,max_bytes=5*2**20),
        gate=dict(main_seconds=1800,audit_seconds=300,max_physics=5900000,max_commands=165,max_bytes=32*2**20),
        eight=dict(main_seconds=900,audit_seconds=180,max_physics=2240000,max_commands=65,max_bytes=2**20))
    p.update(id='T432_OFFLINE_NIGHT_NONLINEAR_RECOVERY_AND_TASK_RETENTION',out=str(OUT),
        data_jobs=[jobs[i] for i in ids],data_proofs=[proofs[i] for i in ids],
        additional_old_jobs=[jobs[i] for i in ids],additional_old_proofs=[proofs[i] for i in ids],
        verified_target_file=str(prior/'data/targets.json'),verified_target_audit=str(prior/'data/audit.json'),
        recovery_targets_path=str(prior/'data/targets.json'),recovery_targets_audit_path=str(prior/'data/audit.json'),
        recovery_target_ids=[r['id'] for r in targets],arm='b',
        stage_budgets=stages,root_limits=dict(max_physics=40000000,max_commands=2000,max_bytes=47*2**20),
        deadline_learning=cuts['learning'],deadline_native=cuts['native'],deadline_all=cuts['all'],
        validation_reserve_seconds=1200,backup_seconds=600,source_seconds=180,
        preflight_discarded_Adam=12,preflight_zero_physics_commands=8,
        maximum_main_wall_seconds=18000,night_max_physics=40000000,
        prior_completed_today_physics=50064905,
        fresh_Adam=True,source_Adam_not_inherited=True,steps=120,
        historical_A_T431_not_repeated=True,
        new_authorization='User explicitlyconfirmedNov9 08:30allstop and independent40Mnewphysics/2000commands/18000mainseconds overnight. Global1GiB/free40 unchanged; not expanding endedT431.',
        pilot='B existingextraMLP4450 only,same59+7actualrecovery/211support as T43166 control. C ifBnotfull329sameinitializer+4450 butworsttrajectoryretentionand3additionaloldhistories, jointengineeringnotisolatedcausal.',
        full_bytes=384*1024,conditional_reserve='Fullworststage plusaudit/finalization/bytes mustfit actualremaining; finitebranches mayfinish early; no repeats/grids.',
        old_training_audit='T422_HISTORY_FAIL_REMAINS_UNRESOLVED',new_RL_steps=0,
        cold_metadata_ledger=str(ROOT/'stage/docs/overnight_20261108/T433_ENCODING_LEDGER.json'))
    assert len(p['initial_success_ids'])==322
    assert sum(s['max_physics'] for s in stages.values())==22290000<=40000000
    assert sum(s['main_seconds'] for s in stages.values())==10500<=18000
    assert sum(s['max_bytes'] for s in stages.values())+2*2**20==p['root_limits']['max_bytes']
    assert 78*p['full_bytes']+2*2**20<stages['gate']['max_bytes']
    fresh_ids={j['id'] for j in p['fresh_jobs']}
    for path in prior.rglob('*.jsonl'):
        if path.name not in ['cases.jsonl','replays.jsonl']:continue
        for line in path.read_text().splitlines():assert json.loads(line).get('job_id') not in fresh_ids
        p['fresh_ledger'][str(path)]=sha(path)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{
        str(prior/'data/targets.json'),str(prior/'data/audit.json'),str(prior/'data/summary.json'),
        str(prior/'learn/audit.json'),str(prior/'learn/summary.json'),str(prior/'learn/development.json'),
        str(ROOT/'stage/runs/t433_cold_metadata/audit.json'),p['cold_metadata_ledger'],
        str(DOC/'PLAN.md'),str(DOC/'METHOD_SOURCES.md')}
    paths|={x['path'] for x in targets}
    p['inputs']={f:sha(f) for f in sorted(paths)}
    used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()+600+180+3600+900+900<cuts['all']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',night_root=p['root_limits'],
       max_main_wall_seconds=18000,global_before=used,full_global_reserved=used+p['root_limits']['max_bytes'],
       stage_budgets=stages,stage_physical_sum=22290000,stages_main_seconds_sum=10500,
       deadlines=cuts,gate_worst_physics=5880004,full384KiB78plusmetadata_fits=True,
       independent_preflight_discarded_Adam=12,zero_physics_commands=8,
       cold_exact_original_bytes_released=11231243,no_unique_model_or_trace_deleted=True))
    print('T432_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for file in HERE.glob('*.py'):
        spec=importlib.util.spec_from_file_location('stage432_check_'+file.stem,file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:
                assert not check(fun.__code__),(file.name,check(fun.__code__))
    for arm in ['b','c']:
        for label in ['A','B']:
            subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t432_contract.py'),'probe',arm,label],cwd=ROOT,check=True,timeout=240)
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t432_contract.py'),'compare',arm],cwd=ROOT,check=True,timeout=60)
    ban_learning()
    t=p['data_jobs'][0]
    counts=mp.get_context('spawn').RawArray('q',5);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children]
        requests=[dict(model=p['student_start_model'],job=t,mode='student',seed=None,cancel_probe=True)]*4
        values=pool.request(requests)
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    geometry=read(p['eight_geometry_path'])
    assert [g['id'] for g in geometry]==[j['id'] for j in p['eight_jobs']]
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=12,
        zero_physics_commands=8,clones_b=read(OUT/'compare_b.json'),clones_c=read(OUT/'compare_c.json'),
        actual_four_workers_cancelled_and_exited=True,eight_geometry_reused_prior_verified56=True,
        no_retained_training_or_export=True,old_T422_history_audit_NOT_repaired=True))
    E.N.manifest(OUT);print('T432_PREFLIGHT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+600+180+3600+900+900<p['deadline_all']
    limit=p['deadline_all']-now-10
    with (DOC/'CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T432',
          '--why=Authorized bounded offline Stage overnight experiment','timeout','--signal=TERM',
          '--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
          cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,
       source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),
       research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,
       six_before_eight=True,seven_skipped=True,new_user_night_authorization=True,
       no_LLM_network_dependency=True,not_a_restart=True))
    print('T432_ACTUALLY_LAUNCHED',child.pid,flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()

