"""Register/preflight/uniquely launch the user-authorized 23:30 Stage window."""
from t431_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess, importlib.util, dis, builtins, types


def prepare():
    assert not OUT.exists()
    DOC.mkdir(exist_ok=True)
    prior=ROOT/'stage/runs/t427_fixed_representation'
    p=copy.deepcopy(read(prior/'protocol.json'))
    summary=read(prior/'learn/summary.json');assert summary['state']=='COMPLETE'
    assert read(prior/'learn/audit.json')['state']=='PASS'
    source=summary['result']['proposed_model'];check=read(summary['result']['check_file'])
    assert check['wide54']['success']==51 and check['old275']['success']==271
    assert check['policy_hash']==source['policy_hash']
    p429=read(ROOT/'stage/runs/t429_feedback_recovery/protocol.json')
    front=next(v for v in p429['recoveries'] if v['job_id']=='dev_task6_21')
    index_path=ROOT/'stage/docs/feedback_20261108/RECOVERY_SUPPORT.json'
    index=read(index_path);assert len(index['successes'])==6
    rows=[json.loads(line) for line in (prior/'learn/cases.jsonl').read_text().splitlines()]
    anchors=[r for r in rows if r['full'] and r['success'] and not r['collision']]
    assert len(anchors)==59
    jobs={j['id']:j for j in [*p['main_jobs'],*p['old275_jobs']]}
    cuts={k:datetime(2026,11,8,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
          for k,h,m in [('learning',22,30),('native',23,0),('all',23,30)]}
    stages=dict(data=dict(main_seconds=480,audit_seconds=120,max_physics=100000,max_commands=10,max_bytes=2*2**20),
                learn=dict(main_seconds=2400,audit_seconds=600,max_physics=7000000,max_commands=340,max_bytes=34*2**20),
                gate=dict(main_seconds=1200,audit_seconds=240,max_physics=3600000,max_commands=100,max_bytes=8*2**20),
                eight=dict(main_seconds=600,audit_seconds=120,max_physics=2240000,max_commands=65,max_bytes=2**20))
    p.update(id='T431_ACTUAL_RECOVERY_SUPPORT_NEW_2330_WINDOW',out=str(OUT),
        student_start_model=source['path'],student_start_sha256=source['sha256'],
        expected_policy_hash=source['policy_hash'],source_model=source,
        current_cases_path=str(prior/'learn/cases.jsonl'),current_anchor_rows=anchors,
        anchor_jobs=[jobs[r['job_id']] for r in anchors],anchor_proofs=anchors,
        initial_success_ids=sorted(r['job_id'] for r in rows if r['success'] and not r['collision']),
        front_recovery=front,gains=p429['gains'],
        recovery_support_index_path=str(index_path),recovery_support_index_sha256=sha(index_path),
        recovery_target_ids=[r['job_id'] for r in index['successes']],
        allowed_teacher_first=[101,76,116,78,73,80,110],
        eight_geometry_path=str(prior/'eight_geometry.json'),steps=120,
        stage_budgets=stages,root_limits=dict(max_physics=13000000,max_commands=700,max_bytes=48*2**20),
        deadline_learning=cuts['learning'],deadline_native=cuts['native'],deadline_all=cuts['all'],
        validation_reserve_seconds=1200,backup_seconds=600,source_seconds=180,
        preflight_zero_physics_commands=8,preflight_discarded_Adam=9,
        prior_today_physics=44726818,whole_day_max_physics=58000000,
        fresh_Adam=True,source_Adam_not_inherited=True,
        new_authorization='UserNov8 explicitlyapprovednewboundedwindow23:30allstopand13Mnewphysics/day<=58M. Not a restart or expansion of any old run.',
        pilot='One fixed front-feedback probe+exact; then fixed120/66 freshAdam actual-recovery support proposal fromcurrentT427researchpolicy, 59anchors+6or7successfultargets. Not a historical singlefactor causalcomparison.',
        maximum_actual_control_recoveries=1,control_parameter_search=False,
        no_teacher_in_PPO=True,new_RL_steps=0,
        conditional_reserve='Every stage full worst remaining physics,bytes,commands,wall and finalization reserve must fit; later gates may be unexecuted null.',
        old_training_audit='T422_FAIL_UNRESOLVED_NOT_REPAIRED_BY_NEW_EXACT_CONTRACTS')
    assert len(p['initial_success_ids'])==322
    assert sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])==6966735
    assert sum(j['case']['step_cap'] for j in p['fresh_jobs']+p['reload_jobs'])==3580002
    assert sum(j['case']['step_cap'] for j in p['eight_jobs'])==2240000
    assert sum(v['max_physics'] for v in stages.values())<=13000000
    assert p['prior_today_physics']+13000000<58000000
    assert sum(v['max_bytes'] for v in stages.values())+3*2**20==p['root_limits']['max_bytes']
    fresh_ids={j['id'] for j in p['fresh_jobs']}
    for folder_name in ['t427_fixed_representation','t428_early_recovery','t429_feedback_recovery']:
        for path in (ROOT/'stage/runs'/folder_name).rglob('*.jsonl'):
            if path.name not in ['cases.jsonl','replays.jsonl']:continue
            for line in path.read_text().splitlines():assert json.loads(line).get('job_id') not in fresh_ids
            p['fresh_ledger'][str(path)]=sha(path)
    paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|set(p['fresh_ledger'])|{
        source['path'],str(prior/'learn/summary.json'),str(prior/'learn/audit.json'),
        summary['result']['check_file'],p['current_cases_path'],p['eight_geometry_path'],str(index_path),
        str(ROOT/'stage/runs/t429_feedback_recovery/protocol.json'),
        str(ROOT/'stage/experiments/t430_full_body_contact/next_feedback.py'),str(DOC/'PLAN.md'),str(DOC/'LITERATURE_AND_DECISION.md')}
    paths|={r['path'] for r in anchors}|{v['path'] for v in index['successes']}|{
        v['source_audit_path'] for v in index['successes']}|{front['reference_path'],front['student_row']['path']}
    # Authenticate original per-episode rows, not just the derived support index.
    paths|={str(Path(v['path']).parents[1]/'cases.jsonl') for v in index['successes']}
    p['inputs']={f:sha(f) for f in sorted(paths)}
    used=E.N.K.S.used(ROOT/'stage/runs')
    assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert time.time()+480+120+1200+600<cuts['learning']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',prior_today_physics=44726818,
        new_window_max_physics=13000000,whole_day_worst=57726818,whole_day_cap=58000000,
        global_before=used,global_full_reserved=used+p['root_limits']['max_bytes'],
        root=p['root_limits'],stages=stages,deadlines=cuts,
        full329_worst_physics=6966735,fresh_and_reload_worst_physics=3580002,
        eight_diagnostic_worst_physics=2240000,preflight_discarded_Adam=9,
        preflight_zero_physics_commands=8,extra_root_metadata_reserve=3*2**20))
    print('T431_REGISTERED_NOT_LAUNCHED',flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for file in HERE.glob('*.py'):
        spec=importlib.util.spec_from_file_location('stage431_check_'+file.stem,file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for item in code.co_consts:
                if isinstance(item,types.CodeType):bad|=check(item)
            return bad
        for fun in vars(module).values():
            if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:
                assert not check(fun.__code__),(file.name,check(fun.__code__))
    for label in ['A','B']:
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t431_contract.py'),'probe',label],cwd=ROOT,check=True,timeout=240)
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'t431_contract.py'),'compare'],cwd=ROOT,check=True,timeout=60)
    ban_learning()
    FRONT.contract_checks()
    t=p['front_recovery'];ref=R.load_reference(t['reference_path'],t['reference_sha256'])
    a=E.N.S.arrays(t['student_row']['path']);i=t['first']
    action,_,_=front_control(ref,a['poses'][i],float(a['states'][i,56]),i*.2,i,p['gains'])
    assert action.shape==(2,) and np.isfinite(action).all() and abs(action).max()<=1
    counts=mp.get_context('spawn').RawArray('q',5);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children]
        requests=[dict(model=p['student_start_model'],job=t['job'],mode='front_feedback',
             first=t['first'],seed=None,cancel_probe=True,reference_path=t['reference_path'],
             reference_sha256=t['reference_sha256'],gains=p['gains'])]*4
        values=pool.request(requests)
        assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    geometry=read(p['eight_geometry_path'])
    assert [g['id'] for g in geometry]==[j['id'] for j in p['eight_jobs']]
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=p['preflight_discarded_Adam'],
        discarded_Adam_in_failed_registration=3,discarded_Adam_in_final_two_clones=6,
        zero_physics_commands=8,clone_comparison=read(OUT/'clone_comparison.json'),
        actual_four_workers_cancelled_and_exited=True,front_feedback_pure_sign_units_and_bounds=True,
        eight_geometry_reused_prior_verified56=True,no_new_geometry_independent_samples=True,
        no_retained_training_or_export=True,old_T422_history_audit_NOT_repaired=True))
    E.N.manifest(OUT);print('T431_PREFLIGHT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    now=time.time();assert now+480+120+1200+600<p['deadline_learning']
    limit=p['deadline_all']-now-10;assert limit>3600
    with (DOC/'CONSOLE.log').open('xb') as log:
        child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T431',
            '--why=User-authorized bounded actual-recovery learning and strict six gates','timeout',
            '--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),
            str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,
        source_remote_commit=receipt['remote_commit'],root_tree=tree,
        protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),
        all_science_offline=True,six_before_eight=True,seven_skipped=True,
        new_user_time_and_physics_authorization=True,not_a_restart=True))
    print('T431_ACTUALLY_LAUNCHED',child.pid,flush=True)


if __name__=='__main__':
    globals()[sys.argv[1]]()
