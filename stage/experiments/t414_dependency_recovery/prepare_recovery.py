"""Engineering recovery of T412 before any navigation-learning update.

Original FAILED directory is immutable. Exact history bytes and all scientific
operators are preserved; repeated physical baselines are explicitly charged.
The original combined budget and absolute deadlines cannot be extended.
"""
from t414_learning import *
import ast,builtins,dis,types,importlib.util

ORIGIN=ROOT/'stage/runs/t412_closed_expert_likelihood'
ORIGIN_SOURCE=ROOT/'stage/experiments/t412_closed_expert_likelihood'

def consumed(p,base):
    paths={p['source_protocol'],p['right_history_file'],p['safe_history_file'],p['initial_model'],p['teacher_model']}
    paths.update(p['left_history_files']);paths.update(p['preflight_query_files'])
    jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job'],*p['main_jobs'],*p['fresh_jobs'],*p['reload_jobs'],*p['centres'],*base['final_old_jobs']]
    paths.update(j['exact_snapshot'] for j in jobs if j.get('exact_snapshot'))
    for f,h in read(ROOT/'stage/manifests/current_working_set_20261107.json')['transitive_hashes'].items():
        assert sha(f)==h;paths.add(f)
    return sorted(paths)

def prepare():
    assert not OUT.exists()
    old=read(ORIGIN/'summary.json');op=read(ORIGIN/'protocol.json');base=read(op['source_protocol'])
    assert old['state']=='FAILED' and old['updates']==dict(live=0,shadow=0) and not old['rounds']
    assert read(ORIGIN/'partial_audit.json')['state']=='PARTIAL_EVIDENCE_PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T412-FAILED.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for name in ['launch.json','driver_started.json','main_started.json','workers.json']:
        v=read(ORIGIN/name)
        for pid in [v[k] for k in ['pid','parent'] if type(v.get(k)) is int]+v.get('workers',[]):assert not Path('/proc',str(pid)).exists(),pid
    assert not (DOC/'T414_CONSOLE.log').exists()
    # Real regression seam: every history consumed by groups must be protected
    # before workers start. The old registration must reproduce the defect.
    required=consumed(op,base);registered={**base['input_hashes'],**op['inputs']}
    omitted=[f for f in required if f not in registered]
    assert op['right_history_file'] in omitted
    assert sha(op['right_history_file'])=='bc43aaa9f5e0e6119fff3c6155546924f648bd522ef3a2d3ceb36b325058093b'
    p=copy.deepcopy(op);origin_bytes=N.K.S.used(ORIGIN);launch=read(ORIGIN/'launch.json')
    original_pref=read(ORIGIN/'preflight.json')['zero_physics_commands'];repair_pref=8
    p.update(id='T414_T412_DEPENDENCY_RECOVERY_NOT_NEW_SCIENCE',max_physics=op['max_physics']-old['physics'],
        max_commands=op['max_commands']-old['commands']-original_pref-repair_pref,
        final_bytes=op['final_bytes']-origin_bytes,minimum_free_bytes=40*2**30,
        main_seconds=4200,learning_seconds=2400,audit_seconds=900,backup_seconds=300,source_seconds=120,all_seconds=6000,
        learning_stop_absolute=launch['learning_deadline'],native_stop_absolute=launch['native_deadline'],all_stop_absolute=launch['all_deadline'])
    p.pop('disk_exception',None)
    p['recovery']=dict(origin_run=str(ORIGIN.relative_to(ROOT)),origin_cases=str(ORIGIN/'cases.jsonl'),origin_frozen_bytes=origin_bytes,
        combined_own_limit_bytes=op['final_bytes'],combined_max_physics=op['max_physics'],combined_max_commands=op['max_commands'],
        original_preflight_commands=original_pref,recovery_preflight_commands=repair_pref,all_preflight_commands=original_pref+repair_pref,
        original_preflight_discarded_Adam=8,recovery_preflight_Adam=0,original_physics=old['physics'],original_commands=old['commands'],
        original_main_wall_s=old['elapsed_s'],original_Adam=0,original_protocol_sha256=sha(ORIGIN/'protocol.json'),
        original_source_unchanged=True,exact_right_history_restored=True,
        baseline_repeat='All14 initial conditions physically repeated and charged; compare all payload/integration/policy hashes to T412, never claim independent new samples.',
        metadata_reserve_bytes=1024**2,original_metadata_reserve_bytes=int(1.5*2**20),
        scientific_changes=[],absolute_deadlines_not_extended=True)
    p['prior_day_actual']=copy.deepcopy(op['prior_day_actual'])
    p['prior_day_actual']['physics']+=old['physics']
    p['prior_day_actual']['commands']+=old['commands']+original_pref+8+repair_pref # T413 factory8 included.
    p['prior_day_actual']['main_wall_s']+=old['elapsed_s']
    files=set(required)|set(op['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files.update(str(f) for f in ORIGIN.rglob('*') if f.is_file() and f.suffix not in ['.lock','.partial','.pyc'])
    files.update([str(receipt),str(DOC/'T412_DEPENDENCY_GREEN.json')])
    p['inputs']={str(f):sha(f) for f in sorted(files)}
    assert not [f for f in required if f not in p['inputs']]
    for f,h in op['inputs'].items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    oldtree=ast.parse((ORIGIN_SOURCE/'t412_learning.py').read_text());newtree=ast.parse((HERE/'t414_learning.py').read_text())
    scientific=['Actor','params','safe_initial','learner','export','atoms','labelled','query_group','groups','nll','old_kl','support_losses','fit_bc','fit_ppo','expert','payload','native_worker','Pool','audit_query']
    for name in scientific:
        a=next(n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        b=next(n for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(a)==ast.dump(b),('Scientific operator changed',name)
    original_worst=read(DOC/'T412_BUDGET.json');worst_bytes=original_worst['worst_bytes']-512*1024
    assert worst_bytes<p['final_bytes'] and origin_bytes+worst_bytes<=op['final_bytes']
    assert original_worst['worst_physics']<=p['max_physics'] and original_worst['main_commands_worst']<=p['max_commands']
    now=time.time();assert now+p['learning_seconds']+120<p['learning_stop_absolute'] and now+p['main_seconds']+120<p['native_stop_absolute'] and now+p['all_seconds']+120<p['all_stop_absolute']
    assert N.K.S.used(ROOT/'stage/runs')+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir()
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T414_BUDGET.json',dict(state='SHARED_ORIGINAL_T412_BUDGET_NOT_RENEWED',origin_bytes=origin_bytes,own_residual=p['final_bytes'],own_worst=worst_bytes,
        combined_own_worst=origin_bytes+worst_bytes,combined_limit=op['final_bytes'],original_physics=old['physics'],remaining_physics=p['max_physics'],
        original_commands=old['commands'],remaining_main_commands=p['max_commands'],old_pref=original_pref,new_pref=repair_pref,
        main_seconds=p['main_seconds'],learning_seconds=p['learning_seconds'],all_seconds=p['all_seconds'],
        original_launch_deadlines={k:launch[k] for k in ['learning_deadline','native_deadline','all_deadline']},unregistered_consumers_fixed=omitted,
        unchanged_scientific_AST=scientific,global_worst=N.K.S.used(ROOT/'stage/runs')+p['final_bytes']))
    print('T414_REGISTERED_SAME_SCIENCE_RESIDUAL_BUDGET',flush=True)

def preflight():
    p=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();base=N.science(p)
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert all(f in p['inputs'] for f in consumed(p,base))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=safe_initial(base,p);teacher=expert(base,p);q.requires_grad_(False);teacher.requires_grad_(False)
    rows=[json.loads(line) for line in (ORIGIN/'cases.jsonl').read_text().splitlines()]
    old=[N.S.arrays(r['path']) for r in rows if r['role'].startswith('initial_old8')]
    g=groups(q,old,p,N.old_data(q,base,lambda:None));g['corrections']=[query_group(q,N.S.arrays(p['preflight_query_files'][0]))]
    with torch.no_grad():losses=[[float(v) for v in support_losses(q,g,p,k)] for k in range(8)]
    assert np.isfinite(losses).all() and max(max(abs(row[0]),abs(row[1])) for row in losses)<1e-26
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage414_actual_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for const in code.co_consts:
                if isinstance(const,types.CodeType):bad|=check(const)
            return bad
        for function in vars(module).values():
            if isinstance(function,types.FunctionType) and function.__module__==module.__name__:assert not check(function.__code__),(name,check(function.__code__))
    set_out(OUT);counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(OUT),native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        ready=pool.ready;ids=[c.pid for c in pool.children]
        result=pool.request([dict(model=str(ORIGIN/'preflight_clone.pt.gz'),job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4)
        assert all(x['state']=='CANCEL_PASS' for x in result) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=0,backward=0,zero_physics_commands=8,
        original_8_synthetic_Adam_reused_not_repeated=True,all_consumed_histories_registered=True,
        real_groups_and_all8_support_losses_finite=True,losses=losses,
        old_loss_machine_roundoff_max=max(max(abs(r[0]),abs(r[1])) for r in losses),not_a_training_threshold_change=True,
        actual_worker_ready_cancel_exit=True,workers=ids,recursive_actual_entry_globals=True,initial_policy_hash=q.phash(),teacher_policy_hash=teacher.phash()))
    N.manifest(OUT);print('T414_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_ADAM',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
