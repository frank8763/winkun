"""Register and execute a bounded demonstration-learning proposal."""
from t365_core import *
import contextlib,ast
import mujoco as mj


def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/runs/t362_narrow_demonstration_learning/protocol.json'))
    assert read(DOC/'T361_AUDIT.json')['state']=='PASS' and read(DOC/'T364_AUDIT.json')['state']=='PASS'
    receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T361.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    s=read(ROOT/'stage/runs/t361_human_terminal_correction/summary.json');row=s['attempts'][0];assert row['success'] and not row['collision'] and row['complete']
    cr=read(ROOT/'stage/runs/t364_early_correction_retry/summary.json')['attempts'][0];assert cr['success'] and cr['complete'] and not cr['collision'] and cr['human_start_decision']==20
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T364.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p.update(experiment='T365_LEARNER_STATE_CORRECTION_AGGREGATION',out=str(OUT),demonstration=row['path'],demonstration_sha256=row['sha256'],
        correction=cr['path'],correction_sha256=cr['sha256'],human_start_decision=20,fit_steps=600,fit_seed=20261105362,actor_lr=.001,arms=['memory'],wall_seconds=2400,final_reserve_seconds=900,audit_seconds=600,all_work_seconds=4200,
        max_commands=97,max_physics=3880000,geometry_commands=1,final_bytes=8*1024**2,full_bytes=768*1024,compact_bytes=8*1024,partial_bytes=768*1024,model_bytes=512*1024,
        max_full_cases=2,max_compact_cases=62,max_live_Adam=600,max_shadow_Adam=600,preflight_discarded_Adam=8,audit_historical_discarded_Adam=600,
        contract=(DOC/'T365_PLAN.md').read_text(),new_RL_updates=0,new_training_native_collection=0)
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{p['demonstration'],str(DOC/'T365_PLAN.md'),str(DOC/'T361_FINDINGS.md'),str(DOC/'T361_AUDIT.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T361.json'),str(ROOT/'stage/experiments/t260_distance_archive/backup.py'),p['correction'],str(DOC/'T364_FINDINGS.md'),str(DOC/'T364_AUDIT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T364.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    worst=2*p['full_bytes']+62*p['compact_bytes']+2*p['model_bytes']+p['partial_bytes']+2*1024**2;assert worst<p['final_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T365_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,
        global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=97,max_physics=3880000,live_supervised_Adam=600,shadow=600,audit_historical=600))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,prior_human_labels=295,correction_labels=277,policy_prefix_labels=0)))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p,0.);data=dataset(p,obj.actor,lambda:None);assert K.C.sha(p['demonstration'])==p['demonstration_sha256']
    mask_report=mask_contract(obj,data);init=obj.state();orig=digest(init);calls=0
    def event(rnd,replica,r):
        with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,r['phase'],r['step']])+'\n')
    for rnd in range(2):
        before=obj.state();shadow=NightLearner(p,0.);shadow.load(before)
        a=fit(p,obj,data,lambda:None,lambda r:event(rnd,'live',r),steps=2);calls+=2
        b=fit(p,shadow,data,lambda:None,lambda r:event(rnd,'shadow',r),steps=2);calls+=2
        assert a==b and obj.hashes()==shadow.hashes()
        if rnd==1:obj.load(before);assert digest(obj.state())==digest(before)
    obj.load(init);assert digest(obj.state())==orig and calls==8 and sum(v.numel() for v in obj.actor.trainable())==6306
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);module.ACTIVE['circle_shift_x_m']=p['training_shift_x_m'];j=p['narrow_jobs'][0]
        try:
            env,obs,_=K.R.task_env(create,j)
            try:
                obs=K.C.perturb_anchor(env,obs,j);geo=P.C.B.geometry(env,obs,w,j['case']['step_cap']);m=env.model
                c=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
                gap=float(m.geom_pos[c,0]-m.geom_size[c,0]-m.geom_pos[wall,0]-m.geom_size[wall,0])
                assert abs(gap-.70)<1e-6 and geo['valid'] and env.step_count==0 and env.data.time==0. and env.max_wheel==5.
                assert np.array_equal(m.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
                assert digest(data[1][0]['x'][0].numpy())==geo['initial120_sha256']
            finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=1,discarded_Adam=calls,human_data_SHA_verified=True,
        actual_demo_initial120_equals_autonomous_original_start=True,mature_Adam_shadow_rollback_exact=True,trainable_parameters=6306,
        initial_demo_error=descriptive_error(obj,data),masked_prefix_contract=mask_report,geometry=geo,capacity=read(DOC/'T365_BUDGET.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=1,discarded_Adam=8)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];learn_end=deadline-900
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
            group=js[offset:offset+2];assert commands+len(group)<=97 and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            assert time.time()+60<deadline;commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,
                learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=readback,stochastic=False))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
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
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T365 native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        js=[*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs'],*p['narrow_jobs']]
        checked=jobs(js,'initial');metrics['initial']=score(checked)
        phase='DATASET';status(True);data=dataset(p,obj.actor,guard);before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=NightLearner(p,0.);shadow.load(before)
        err0=descriptive_error(obj,data);save(OUT/'data_contract.json',dict(old_sequences=211,human_labels_total=572,prior_human_labels=295,correction_human_labels=277,masked_learner_prefix=20,full_unroll_correction_states=297,
            human_sequence_sha256=p['demonstration_sha256'],correction_sha256=p['correction_sha256'],schedule_sha256=digest(schedule(p,data[0])),trainable_parameters=6306,initial_error=err0,new_RL_updates=0))
        phase='SUPERVISED_FULL_HISTORY';status(True)
        try:
            trace=fit(p,obj,data,guard,lambda r:event('live',r));other=fit(p,shadow,data,guard,lambda r:event('shadow',r))
            assert trace==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
        except BaseException:
            obj.load(before);save(OUT/'interrupted.json',dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
        save(OUT/'trace.json',trace);block=dict(input_hashes=input_hashes,output_hashes=obj.hashes(),live_Adam=600,shadow_Adam=600,exact_shadow=True,trace_sha256=digest(trace))
        save(OUT/'block.json',block);metrics['fit_error']=descriptive_error(obj,data);model=export(p,obj,OUT/'candidate.pt.gz');save(OUT/'model.json',model)
        final=jobs(js,'final');metrics['final']=score(final)
        jobs(js,'model_readback',{v['row']['job_id']:v for v in final},model['path']);state='COMPLETE'
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
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T365 audit600s')));signal.alarm(600)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);obj=NightLearner(p,0.);data=dataset(p,obj.actor,lambda:None)
    assert K.C.sha(p['correction'])==p['correction_sha256'];assert mask_contract(obj,data)['state']=='PASS'
    b=read(OUT/'block.json');assert obj.hashes()==b['input_hashes'];ev=[];trace=fit(p,obj,data,lambda:None,lambda r:ev.append(r))
    assert trace==read(OUT/'trace.json') and digest(trace)==b['trace_sha256'] and obj.hashes()==b['output_hashes']
    events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:assert [dict(phase=v[1],step=v[2]) for v in events if v[0]==replica]==ev
    model=s['model'];assert K.C.sha(model['path'])==model['sha256'];z=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
    assert digest(z['state'])==digest(obj.state());rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==64 and len(reps)==32;physics=0;full=0
    init_q=NightActor(p,0.).eval().requires_grad_(False)
    for r in rows:
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
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==97 and s['updates']==dict(live=600,shadow=600) and full==2 and s['elapsed_s']<=2400
    result=dict(state='PASS',new_native=0,historical_discarded_Adam=600,source_model_data_SHA_exact=True,live_shadow_complete_reconstruction=True,
        human_labels_not_onpolicy=True,masked_prefix_contract=mask_contract(NightLearner(p,0.),data),full_current_history_reward_native_model_readback_exact=True,commands=97,physics_steps=physics,result=s['result'],stable_original_six=False)
    save(OUT/'audit.json',result);save(DOC/'T365_AUDIT.json',result);signal.alarm(0);print(json.dumps(dict(state='AUDIT_PASS',historical_Adam=600,native=0)))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T365_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage narrow human demonstration learning',
        'timeout','--signal=TERM','--kill-after=1s','4200s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+2400,all_work_deadline_epoch=epoch+4200,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T365_DRIVER.json').exists();active=None
    save(DOC/'T365_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='NATIVE'))
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
    def stop(*_):raise TimeoutError('Bounded T365 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'run.py'),'main'],min(2400,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T365_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'run.py'),'audit'],min(600,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T365_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T365'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T365_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T365.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T365_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T365.json')))
    except BaseException:
        save(DOC/'T365_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

