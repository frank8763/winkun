from t372_core import *
import contextlib,ast
import mujoco as mj


def baseline_rows(p):
    assert K.C.sha(p['baseline_cases_path'])==p['baseline_cases_sha256']
    rows=[r for r in read(p['baseline_cases_path']) if r['role']=='initial']
    assert len(rows)==32 and len({r['job_id'] for r in rows})==32
    for r in rows:
        assert r['policy_hash']==p['repair_parent']['policy_hash'] and r['array_hashes']==p['initial_expected_hashes'][r['job_id']]
        assert K.C.sha(r['path'])==r['sha256'] and r['complete'] and not r['administrative_censor']
    return rows


def prepare():
    assert (not OUT.exists() or ((OUT/'preflight_failed_001/failure.json').exists() and not (OUT/'preflight.json').exists() and not (OUT/'launch.json').exists() and not (OUT/'status.json').exists())) and read(DOC/'T371_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE_WITH_TRANSPORT_RECOVERY'
    base=ROOT/'stage/runs/t371_terminal_head_repair'
    assert read(base/'audit.json')['state']=='PASS' and read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T371.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(base/'protocol.json'))
    p.update(id='T372',folder=OUT.name,out=str(OUT),experiment='T372_NONLINEAR_STEERING_LOGIT_REPAIR',adapter_seed=20261105372,
        arms=['nonlinear_repair'],wall_seconds=1800,final_reserve_seconds=600,audit_seconds=600,all_work_seconds=3600,
        max_commands=66,max_physics=2640000,final_bytes=int(6.5*1024**2),max_full_cases=2,max_compact_cases=31,
        baseline_cases_path=str(base/'cases.json'),baseline_cases_sha256=K.C.sha(base/'cases.json'),historical_initial_cases=32,fresh_initial_cases=1,
        required_recovery_ids=['narrow070_dev_0','narrow070_dev_2','narrow070_dev_3'],contract=(DOC/'T372_PLAN.md').read_text())
    torch.set_num_threads(1);parent=PRIOR.OLD.NightActor(p,0.);parent.load_learned(read_model(p['repair_parent']['path']));parent.requires_grad_(False)
    with torch.no_grad():
        source=arrays(p['repair_source']['path']);x=torch.tensor(source['states']);z=parent.encoder(((x.double()-parent.mean)/parent.scale).clamp(-10,10));mean_h=parent.core(z[:,None])[0][248:253,0].mean(0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed']);layer=torch.nn.Linear(16,4).double();p['adapter_initial_bias']=(-layer.weight@mean_h).tolist()
    p['initialization']='Fixed-seed A; a=-A*mean_training_repair_hidden; v=b=0; no target/seed search'
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(f) for f in (HERE/'preflight_failed_001').iterdir()}|{str(DOC/'T372_PREFLIGHT_INITIALIZATION_FAILURE.json'),str(DOC/'T372_PLAN.md'),str(DOC/'T372_NONLINEAR_REPAIR_DRAFT.md'),str(DOC/'T371_CLOSEOUT.json'),str(DOC/'T371_FINDINGS.md'),str(base/'cases.json'),str(base/'audit.json'),str(base/'summary.json'),str(base/'episodes/initial_narrow070_dev_0.npz'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T371.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};baseline_rows(p);used=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)
    worst=2*p['full_bytes']+31*p['compact_bytes']+2*p['model_bytes']+3*p['partial_bytes']+1024**2+128*1024+128*1024
    assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir(exist_ok=True);(OUT/'episodes').mkdir(exist_ok=True);save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T372_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=66,max_physics=2640000,live_Adam=300,shadow_Adam=300,historical_audit_Adam=300,trainable_parameters=73))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,full_global_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);report=repair_contract(obj,data);initial=obj.state();calls=0;fixed=fixed_hash(obj.actor)
    for rnd in range(2):
        before=obj.state();immutable=digest(before);shadow=NightLearner(p);shadow.load(before)
        def event(replica,r):
            with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,r['phase'],r['step']])+'\n')
        a=fit(p,obj,data,lambda:None,lambda r:event('live',r),steps=2);calls+=2
        b=fit(p,shadow,data,lambda:None,lambda r:event('shadow',r),steps=2);calls+=2
        assert a==b and obj.hashes()==shadow.hashes() and digest(before)==immutable and fixed_hash(obj.actor)==fixed
        assert any(r['adapter_feature_gradient_nonzero'] for r in a)
        if rnd==1:obj.load(before);assert digest(obj.state())==immutable
    obj.load(initial);assert digest(obj.state())==digest(initial) and calls==8
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);module.ACTIVE['circle_shift_x_m']=p['narrow_jobs'][0]['circle_shift_x_m'];env,obs,_=K.R.task_env(create,p['narrow_jobs'][0])
        try:
            obs=K.C.perturb_anchor(env,obs,p['narrow_jobs'][0]);g=P.C.B.geometry(env,obs,w,40000);assert g['valid'] and env.step_count==0 and env.data.time==0. and env.max_wheel==5.
            assert np.array_equal(np.asarray(g['full120'],np.float32),data[1]['x'][0].numpy()) and fixed_hash(obj.actor)==p['repair_parent']['hashes']['learned']
        finally:w.events.pop(id(env),None);env.close();w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=1,discarded_Adam=calls,repair_contract=report,mature_shadow_and_rollback_exact=True,all_parent_bytes_immutable=True,geometry=g,budget=read(DOC/'T372_BUDGET.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=1,discarded_Adam=8)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];learn_end=deadline-600
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2);processes=[];pipes=[];rows=[];replays=[];commands=1;active=[];phase='STARTING';last=0.;state='FAILED';model=None;metrics={};block=None
    obj=NightLearner(p,0.);initial=obj.state();updates=dict(live=0,shadow=0);meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,
            commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    def guard():assert time.time()<learn_end;status();meter.check()
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def jobs(js,role,reference=None,readback=None):
        nonlocal commands,phase,active
        phase=role;items=[]
        for offset in range(0,len(js),2):
            group=js[offset:offset+2];assert commands+len(group)<=66 and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            assert time.time()+60<deadline;commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,
                learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=readback,stochastic=False))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if role=='initial':assert r['array_hashes']==p['initial_expected_hashes'][r['job_id']]
                if not r['complete']:
                    b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Censored partial retained, no retry')
                if reference is not None:
                    ref=reference[r['job_id']];assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                    assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r);save(OUT/'replays.json',replays)
                else:
                    full=r['job_id']=='narrow070_dev_0';data=packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']}
                    b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();assert len(blob)<=p['full_bytes' if full else 'compact_bytes']
                    path=OUT/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                    r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full,saved_array_hashes={k:digest(v) for k,v in data.items()})
                    rows.append(r);save(OUT/'cases.json',rows)
                items.append(item)
            status(True)
        return items
    def score(items):
        return {name:{k:sum(bool(v['row'][k]) for v in items[sl]) for k in ['success','collision','timeout']} for name,sl in [('old20',slice(0,20)),('wide5',slice(20,25)),('original2',slice(25,27)),('new0705',slice(27,32))]}
    def event(replica,r):
        if r['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([replica,r['phase'],r['step']])+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T372 native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        js=[*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs'],*p['narrow_jobs']]
        checked=[dict(row=r) for r in baseline_rows(p)];metrics['initial']=score(checked)
        metrics['initial_baseline_is_historical']=True;metrics['historical_initial_cases']=32;metrics['fresh_initial_cases']=1
        jobs([p['narrow_jobs'][0]],'initial')
        phase='DATASET';status(True);data=dataset(p,obj.actor,guard);before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=NightLearner(p,0.);shadow.load(before)
        fallback=export(p,obj,OUT/'initial_fallback.pt.gz')
        err0=descriptive_error(obj,data);save(OUT/'data_contract.json',dict(old_sequences=211,repair_source_sha256=p['repair_source']['sha256'],explicit_nonhuman_override_labels=5,effective_steer_parameters=73,repair_contract=repair_contract(obj,data),schedule_sha256=digest(schedule(p,data[0])),effective_trainable_parameters=73,initial_error=err0,new_RL_updates=0,fallback=fallback))
        phase='SUPERVISED_FULL_HISTORY';status(True)
        try:
            trace=fit(p,obj,data,guard,lambda r:event('live',r));other=fit(p,shadow,data,guard,lambda r:event('shadow',r))
            assert trace==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
        except BaseException:
            obj.load(before);save(OUT/'interrupted.json',dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
        save(OUT/'trace.json',trace);block=dict(input_hashes=input_hashes,output_hashes=obj.hashes(),live_Adam=300,shadow_Adam=300,exact_shadow=True,trace_sha256=digest(trace))
        save(OUT/'block.json',block);metrics['fit_error']=descriptive_error(obj,data);model=export(p,obj,OUT/'candidate.pt.gz');save(OUT/'model.json',model)
        final=jobs(js,'final');metrics['final']=score(final)
        jobs(js,'model_readback',{v['row']['job_id']:v for v in final},model['path'])
        before_ids={v['row']['job_id'] for v in checked if v['row']['success']};after_ids={v['row']['job_id'] for v in final if v['row']['success']}
        metrics['retention_pass']=before_ids<=after_ids;metrics['required_recovery_pass']=set(p['required_recovery_ids'])<=after_ids
        metrics['research_only_rejected']=not(metrics['retention_pass'] and metrics['required_recovery_pass']);metrics['initial_success_ids']=sorted(before_ids);metrics['final_success_ids']=sorted(after_ids);state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    if model is None:obj.load(initial);model=export(p,obj,OUT/'initial_or_rollback.pt.gz')
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,model=model,result=metrics,
        primary_cases=len(rows),native_model_readbacks=len(replays),all_workers_stopped=all(not v.is_alive() for v in processes),new_RL_updates=0,new_native_training_collection=0,
        research_candidate_not_promoted=True,old275=None,six54=None,fresh16=None,reload78=None,stable_original_six=False,entered7or8=False)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T372 audit600s')));signal.alarm(600)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);obj=NightLearner(p,0.);data=dataset(p,obj.actor,lambda:None)
    assert repair_contract(obj,data)['state']=='PASS'
    fallback=read(OUT/'data_contract.json')['fallback'];assert K.C.sha(fallback['path'])==fallback['sha256']
    fallback_state=torch.load(io.BytesIO(gzip.decompress(Path(fallback['path']).read_bytes())),weights_only=False,map_location='cpu')['state']
    assert digest(fallback_state)==digest(obj.state())
    b=read(OUT/'block.json');assert obj.hashes()==b['input_hashes'];ev=[];trace=fit(p,obj,data,lambda:None,lambda r:ev.append(r))
    assert trace==read(OUT/'trace.json') and digest(trace)==b['trace_sha256'] and obj.hashes()==b['output_hashes']
    events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:assert [dict(phase=v[1],step=v[2]) for v in events if v[0]==replica]==ev
    model=s['model'];assert K.C.sha(model['path'])==model['sha256'];z=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
    assert digest(z['state'])==digest(obj.state()) and fixed_hash(obj.actor)==fixed_hash(NightActor(p));rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==33 and len(reps)==32;physics=0;full=0
    init_q=NightActor(p,0.).eval().requires_grad_(False)
    for r in rows:
        if r['role']=='initial':assert r['array_hashes']==p['initial_expected_hashes'][r['job_id']]
        assert K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as f:assert {k:digest(f[k]) for k in f.files}==r['saved_array_hashes']
        a=arrays(r['path']);assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'];physics+=r['physics_steps']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any();c,g,t=map(bool,a['end_flags'][-1])
        assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout'] and not r['administrative_censor']
        if not r['full_arrays_saved']:continue
        full+=1;assert {k:digest(v) for k,v in a.items()}==r['array_hashes'];q=init_q if r['role']=='initial' else obj.actor
        q.reset_noise();q.sample=False;h=None;assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        for k,x in enumerate(a['states']):
            before=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(before,a['hidden_before'][k])
            with torch.no_grad():action,loc,h,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k])
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);disc=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            rw=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not(c or g or t))
            if c or g or t:rw+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(rw-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
    for r in reps:
        ref=next(v for v in rows if v['role']=='final' and v['job_id']==r['job_id'])
        assert ref['array_hashes']==r['array_hashes'] and ref['integration_sha256']==r['integration_sha256'] and r['all_arrays_hidden_history_integration_exact'] and not r['independent_sample'];physics+=r['physics_steps']
    before={r['job_id'] for r in baseline_rows(p) if r['success']};after={r['job_id'] for r in rows if r['role']=='final' and r['success']}
    assert s['result']['retention_pass']==(before<=after)
    assert s['result']['required_recovery_pass']==(set(p['required_recovery_ids'])<=after)
    assert s['result']['research_only_rejected']==(not(s['result']['retention_pass'] and s['result']['required_recovery_pass']))
    assert s['result']['initial_baseline_is_historical'] and s['result']['historical_initial_cases']==32 and s['result']['fresh_initial_cases']==1
    assert s['result']['initial_success_ids']==sorted(before) and s['result']['final_success_ids']==sorted(after)
    groups={'old20':{j['id'] for j in p['old_jobs']},'wide5':{j['id'] for j in p['course_dev_jobs']},'original2':{j['id'] for j in p['original_jobs']},'new0705':{j['id'] for j in p['narrow_jobs']}}
    for role in ['initial','final']:
        for group,ids in groups.items():
            selected=[r for r in (baseline_rows(p) if role=='initial' else rows) if r['job_id'] in ids and (role=='initial' or r['role']=='final')]
            assert len(selected)==len(ids)
            assert s['result'][role][group]=={k:sum(bool(r[k]) for r in selected) for k in ['success','collision','timeout']}
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==66 and s['updates']==dict(live=300,shadow=300) and full==2 and s['elapsed_s']<=1800
    result=dict(state='PASS',new_native=0,historical_discarded_Adam=300,source_model_data_SHA_exact=True,live_shadow_complete_reconstruction=True,
        nonhuman_intervention_labels_not_onpolicy=True,nonlinear_repair_contract=repair_contract(NightLearner(p,0.),data),full_current_history_reward_native_model_readback_exact=True,commands=66,physics_steps=physics,result=s['result'],stable_original_six=False)
    save(OUT/'audit.json',result);save(DOC/'T372_AUDIT.json',result);signal.alarm(0);print(json.dumps(dict(state='AUDIT_PASS',historical_Adam=300,native=0)))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T372_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage learned terminal steering repair',
        'timeout','--signal=TERM','--kill-after=1s','3600s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+1800,all_work_deadline_epoch=epoch+3600,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T372_DRIVER.json').exists();active=None
    save(DOC/'T372_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='NATIVE'))
    def call(args,seconds,log):
        nonlocal active
        with log.open('xb') as f:
            active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),*args],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=active.wait(timeout=seconds)
                if rc:raise subprocess.CalledProcessError(rc,str(args))
            except BaseException:
                # Reap the complete subprocess group even if its leader exited.
                try:os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                if active.poll() is None:
                    try:active.wait(1)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.wait();raise
            finally:active=None
    def stop(*_):raise TimeoutError('Bounded T372 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'run.py'),'main'],min(1800,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T372_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'run.py'),'audit'],min(600,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T372_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(HERE/'transport.py'),'T372'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T372_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T372.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T372_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T372.json')))
    except BaseException:
        save(DOC/'T372_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

