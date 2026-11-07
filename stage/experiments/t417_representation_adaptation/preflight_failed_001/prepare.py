"""Register one representation-adaptation test; no prepare/launch at import.

The audited T414 round0 is the historical residual-only comparator. It is not
rerun, counted as fresh data, or used as evidence that PPO was attempted.
"""
from t417_learning import *
import ast,builtins,dis,types,importlib.util
from datetime import datetime
from zoneinfo import ZoneInfo
ORIGIN=ROOT/'stage/runs/t414_dependency_recovery'

def consumed(p,base):
    paths={p['source_protocol'],p['right_history_file'],p['safe_history_file'],p['initial_model'],p['teacher_model']}
    paths.update(p['left_history_files']);paths.update(p['preflight_query_files'])
    jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job'],*p['main_jobs'],*p['fresh_jobs'],*p['reload_jobs'],*p['centres'],*base['final_old_jobs']]
    paths.update(j['exact_snapshot'] for j in jobs if j.get('exact_snapshot'))
    for f,h in read(ROOT/'stage/manifests/current_working_set_20261107.json')['transitive_hashes'].items():
        assert sha(f)==h;paths.add(f)
    h=p['historical_control'];paths.update(h[k] for k in ['summary','audit','cases','round_record','protocol'])
    paths.update(item['row']['path'] for item in h['queries'])
    return sorted(paths)

def prepare():
    assert not OUT.exists() and not (DOC/'T417_CONSOLE.log').exists()
    original=read(ORIGIN/'summary.json');assert original['state']=='COMPLETE'
    assert read(ORIGIN/'audit.json')['state']=='PASS'
    assert read(DOC/'T416_FINAL_REPORT.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T414.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for name in ['launch.json','driver_started.json','main_started.json','workers.json']:
        v=read(ORIGIN/name)
        for pid in [v[k] for k in ['pid','parent'] if type(v.get(k)) is int]+v.get('workers',[]):assert not Path('/proc',str(pid)).exists(),pid
    p=copy.deepcopy(read(ORIGIN/'protocol.json'));base=read(p['source_protocol']);p.pop('recovery',None);p.pop('disk_exception',None)
    prior=copy.deepcopy(p['prior_day_actual']);prior['physics']+=original['physics'];prior['commands']+=original['commands'];prior['main_wall_s']+=original['elapsed_s']
    # Existing prior ledger already includes T414's 8 preflight zero-physical commands.
    tz=ZoneInfo('Asia/Shanghai')
    p.update(id='T417_STUDENT_REPRESENTATION_ADAPTATION_HISTORICAL_T414_CONTROL',
        final_bytes=18*2**20,max_physics=16_000_000,max_commands=600,main_seconds=2400,learning_seconds=1200,
        audit_seconds=900,backup_seconds=300,source_seconds=120,all_seconds=4200,minimum_free_bytes=40*2**30,
        learning_stop_absolute=datetime(2026,11,7,18,45,tzinfo=tz).timestamp(),
        native_stop_absolute=datetime(2026,11,7,19,30,tzinfo=tz).timestamp(),
        all_stop_absolute=datetime(2026,11,7,20,0,tzinfo=tz).timestamp(),
        query_pairs=[[0,1]],bc_steps=400,epochs=8,minimum_cycle_seconds=600,prior_day_actual=prior,
        scientific_factor='Only additional trainable student encoder/core; same safe function, extra, data, losses, optimizer, seed and inference topology. Base/head/adapter frozen.',
        live_Adam_max=408,shadow_Adam_max=408,independent_historical_Adam_max=408,preflight_discarded_Adam=8,
        preflight_zero_physics_commands=8,preflight_native=0,new_teacher_rollouts=0,
        historical_comparator='T414 round0 existing 400-step residual-only proposal; not a fresh matched-seed repeat or parameter-count-controlled result.',
        bc_admission='old8 and RIGHT all GOAL, prior success IDs retained, LEFT at least1/5; otherwise whole Actor+Adam rollback. No relaxed six-metre gate.',
        final_full_admission='same retained candidate final14 all GOAL then old275+wide54 all GOAL then unconsumed16fresh then78 exact independentnative reload.')
    record=read(ORIGIN/'blocks/0/result.json');assert record==original['rounds'][0] and len(record['queries'])==2 and not record['bc']['accepted']
    queries=copy.deepcopy(record['queries']);assert all(q['row']['success'] and q['row']['full'] for q in queries)
    p['historical_control']=dict(summary=str(ORIGIN/'summary.json'),audit=str(ORIGIN/'audit.json'),cases=str(ORIGIN/'cases.jsonl'),
        round_record=str(ORIGIN/'blocks/0/result.json'),protocol=str(ORIGIN/'protocol.json'),queries=queries,
        A_checks=record['bc']['checks'],A_Adam_live=400,A_repeated_Adam=0)
    p['preflight_query_files']=[q['row']['path'] for q in queries]
    # T414 inputs already cover all old211 datasets and indirect simulator/model sources.
    files=set(p['inputs'])|set(consumed(p,base))|{str(f) for f in HERE.glob('*.py')}
    files.update([str(receipt),str(DOC/'T416_FINAL_REPORT.json')])
    oldrows=[json.loads(line) for line in (ORIGIN/'cases.jsonl').read_text().splitlines()]
    assert len(oldrows[:14])==14
    files.update(row['path'] for row in oldrows[:14])
    p['inputs']={str(f):sha(f) for f in sorted(files)}
    assert all(f in p['inputs'] for f in consumed(p,base))
    # Absolute worst-case includes all conditional full-gate branches and both
    # exact native replays; compact results are at most5KiB each.
    jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job']]
    worst_jobs=[*jobs,*jobs,*jobs,*([p['left5_jobs'][0]]*8),*jobs,*p['centres'],*p['centres'],*p['main_jobs'],*base['final_old_jobs'],*p['fresh_jobs'],*p['reload_jobs']]
    worst_commands=4+len(worst_jobs);worst_physics=sum(j['case']['step_cap'] for j in worst_jobs)
    assert worst_commands+8<=p['max_commands'] and worst_physics<=p['max_physics']
    worst_bytes=8*p['full_bytes']+4*p['full_bytes']+len(worst_jobs)*5*1024+5*1024**2+2*1024**2
    assert worst_bytes<=p['final_bytes']
    assert prior['physics']+p['max_physics']<=50_000_000 and prior['commands']+p['max_commands']<=2600 and prior['main_wall_s']+p['main_seconds']<=20000
    now=time.time();assert now+1200+60<p['learning_stop_absolute'] and now+2402<p['native_stop_absolute'] and now+4200<p['all_stop_absolute']
    assert N.K.S.used(ROOT/'stage/runs')+p['final_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T417_BUDGET.json',dict(state='NEW_REPRESENTATION_FACTOR_BOUNDED',own_bytes=p['final_bytes'],worst_bytes=worst_bytes,
        max_commands=p['max_commands'],worst_main_commands=worst_commands,preflight_zero_native_commands=8,
        max_physics=p['max_physics'],worst_physics=worst_physics,prior_day_actual=prior,
        day_physics_worst=prior['physics']+p['max_physics'],day_commands_worst=prior['commands']+p['max_commands'],
        day_main_wall_worst=prior['main_wall_s']+p['main_seconds'],global_worst=N.K.S.used(ROOT/'stage/runs')+p['final_bytes'],
        historical_A_repeated_updates=0,new_teacher_rollouts=0,live_max=408,shadow_max=408,audit_discarded_max=408,preflight_discarded=8))
    print('T417_REGISTERED_NOT_LAUNCHED',flush=True)

def recursive_globals():
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage417_actual_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
            for const in code.co_consts:
                if isinstance(const,types.CodeType):bad|=check(const)
            return bad
        for function in vars(module).values():
            if isinstance(function,types.FunctionType) and function.__module__==module.__name__:assert not check(function.__code__),(name,check(function.__code__))

def preflight():
    p=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();base=N.science(p)
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert all(f in p['inputs'] for f in consumed(p,base));recursive_globals()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);initial=q.phash();assert initial=='8182a08428acebd871fc1d42d4dcd52a3b995400207e75e5c84f5ec206b1d930'
    rows=[json.loads(line) for line in Path(p['historical_control']['cases']).read_text().splitlines()]
    old=[N.S.arrays(r['path']) for r in rows if r['role'].startswith('initial_old8')]
    g=groups(q,old,p,N.old_data(q,base,lambda:None));g['corrections']=[]
    for query in p['historical_control']['queries']:
        a=N.S.arrays(query['row']['path']);audit_query(q,a,query['case'],p);g['corrections'].append(query_group(q,a))
    with torch.no_grad():
        losses=[[float(v) for v in support_losses(q,g,p,k)] for k in range(8)]
        for a in old:
            x=torch.tensor(a['states']);locations=q.sequence(x,q.base_actions(x));assert torch.max(torch.abs(locations-torch.tensor(a['locations'])))<1e-10
            hidden=None;seq=[]
            for row in x:
                _,loc,hidden,_=q.step(row[None],hidden);seq.append(loc[0])
            assert torch.max(torch.abs(locations-torch.stack(seq)))<1e-10
    assert np.isfinite(losses).all() and max(max(abs(row[0]),abs(row[1])) for row in losses)<1e-26
    fixed=original_hash(q);a,ao=learner(base,p);b,bo=learner(base,p);clones=[]
    # Four synthetic sequence steps per clone, eight discarded Adam total.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(417);x=torch.randn((24,120),dtype=torch.float64)*.1
    carry_checks=[]
    for student,optimizer in [(a,ao),(b,bo)]:
        snaps=[]
        for k in range(4):
            optimizer.zero_grad(set_to_none=True);base_actions=student.base_actions(x)
            pred=student.sequence(x,base_actions);target=base_actions+.03*torch.sin(torch.arange(24,dtype=torch.float64))[:,None]
            loss=(pred-target).square().mean();loss.backward()
            assert all(v.grad is None for name,v in student.named_parameters() if not name.startswith(TRAINED_PREFIXES))
            if k>=2:
                for prefix in ['encoder.','core.']:
                    assert any(v.grad is not None and bool(torch.count_nonzero(v.grad)) for name,v in student.named_parameters() if name.startswith(prefix)),prefix
            optimizer.step();snaps.append(L.state(student,optimizer));assert original_hash(student)==fixed
        assert student.phash()!=initial and snaps[1]['optimizer']['state']
        # A nonzero learned readout is essential: the initial safe zero
        # readouts make location equality independent of recurrent state.
        # GRU batch_first=False: (N,1,D) is N time steps, one batch.
        assert student.core.batch_first is False
        with torch.no_grad():
            actual_x=torch.tensor(old[0]['states']);student.reset_noise();hidden=None;locations=[];hidden_rows=[]
            for row in actual_x:
                _,location,hidden,_=student.step(row[None],hidden);locations.append(location[0]);hidden_rows.append(hidden[0,0])
            whole,final_hidden=student.core(student.encoder(student.normalized(actual_x))[:,None])
            location_error=float((student.sequence(actual_x,student.base_actions(actual_x))-torch.stack(locations)).abs().max())
            hidden_error=float((whole[:,0]-torch.stack(hidden_rows)).abs().max())
            assert location_error<1e-10 and hidden_error<1e-10 and float((final_hidden-hidden).abs().max())<1e-10
            carry_checks.append(dict(nonzero_readout=True,batch_first=False,location_error=location_error,hidden_error=hidden_error))
        completed=L.statehash(student,optimizer);L.load(student,optimizer,snaps[1]);assert digest(L.state(student,optimizer))==digest(snaps[1])
        try:raise L.StageStop('synthetic interruption rollback')
        except L.StageStop:L.load(student,optimizer,snaps[1])
        assert digest(L.state(student,optimizer))==digest(snaps[1]);clones.append((completed,L.statehash(student,optimizer)))
    assert clones[0]==clones[1]
    set_out(OUT);model=export(q,opt,OUT/'preflight_clone.pt.gz');counts=mp.get_context('spawn').RawArray('q',5)
    pool=Pool(dict(p,out=str(OUT),native_stop_epoch=time.time()+180),counts,lambda:None)
    try:
        ids=[c.pid for c in pool.children];ready=pool.ready
        result=pool.request([dict(model=model['path'],job=p['right_job'],first=0,seed=None,cancel_probe=True)]*4)
        assert all(v['state']=='CANCEL_PASS' for v in result) and sum(counts)==0
    finally:pool.close()
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    save(OUT/'preflight.json',dict(state='PASS',native=0,discarded_Adam=8,zero_physics_commands=8,
        actual_worker_ready_cancel_exit=True,workers=ids,recursive_actual_entry_globals=True,
        safe_parent_function_hash=initial,trainable_prefixes=TRAINED_PREFIXES,
        trainable_parameters=sum(v.numel() for v in params(q)),all_actual_consumers_SHA=True,
        old_support_losses_finite=True,old_loss_roundoff_max=max(max(abs(r[0]),abs(r[1])) for r in losses),
        full_causal_sequence_equals_step_hidden_carry=True,nonzero_updated_clone_carry=carry_checks,clone_Adam_and_mature_rejection_interruption_rollback_exact=True,
        historical_teacher_queries_verified_without_physics=True,historical_A_Adam_not_repeated=True))
    N.manifest(OUT);print('T417_PREFLIGHT_PASS_0NATIVE_8DISCARDED_ADAM',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
