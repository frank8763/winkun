from sequence_value import *
import argparse,builtins,multiprocessing as mp,signal,subprocess,symtable,contextlib

def preflight():
    assert not any((OUT/f).exists() for f in ['preflight.json','status.json','launch.json','preflight_failure.json'])
    p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);q=B.actor(p)
    torch.manual_seed(p['seed']);model=SequenceCritic();rng=np.random.default_rng(p['seed']);x=torch.tensor(rng.normal(size=(64,120)),dtype=torch.float32);a=torch.tensor(rng.normal(size=(64,6)),dtype=torch.float32)
    data=(x,a,torch.tensor(rng.normal(size=(64,64,2)),dtype=torch.float32),torch.zeros((64,64),dtype=torch.long),torch.eye(64,dtype=torch.bool),torch.arange(64))
    loss,_,_=objective(model,data,'sequence');loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    changed=(x,a+torch.tensor([0,0,1,1,1,1]),*data[2:]);assert torch.equal(logits(model,data,'single'),logits(model,changed,'single'))
    assert torch.equal(logits(model,data,'blind'),logits(model,(x,a+1,*data[2:]),'blind'))
    teacher=copy.deepcopy(model).requires_grad_(False);student=copy.deepcopy(model);student.query.requires_grad_(False);teacher_hash=S.digest(teacher.state_dict())
    target=logits(teacher,data,'sequence').detach();pred=logits(student,data,'distilled');delta=target-pred;dl=(torch.where(delta>=0,p['expectile'],1-p['expectile'])*delta.square()).mean();dl.backward()
    assert all(v.grad is None for v in teacher.parameters()) and all(v.grad is None for v in student.query.parameters())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in student.state_action.parameters());assert S.digest(teacher.state_dict())==teacher_hash
    for module in [sys.modules[__name__],sys.modules['sequence_value']]:
        tab=symtable.symtable(Path(module.__file__).read_text(),str(module.__file__),'exec');missing=set()
        def walk(t):
            for s in t.get_symbols():
                if s.is_referenced() and s.is_global() and s.get_name() not in vars(module) and not hasattr(builtins,s.get_name()):missing.add(s.get_name())
            for ch in t.get_children():walk(ch)
        walk(tab);assert not missing,(module.__name__,missing)
    witness=None;log=None;geometry=[];original=R.task_env;lock=None
    try:
        lock=Budget(ROOT,OUT,p);folder=OUT/'preflight_native';folder.mkdir();log=LimitedLog(OUT/'preflight_native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(folder)
        spec=importlib.util.spec_from_file_location('t309_preflight_history',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base)
        for goal in p['task_goals']:
            j=next(j for j in p['jobs'] if j['source_reset_id']==goal['id'] and j['arm']=='center');env=None
            try:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):env,obs,_=original(create,j)
                ih=R.arrays_hash(R.integration(env));obs=witness.alternate(env,j['active_goal_xy'],witness.get(env));assert ih==R.arrays_hash(R.integration(env));obs=C.perturb_anchor(env,obs,j)
                assert not env._collision() and float(env._goal_state()[1])>.2
                meta=witness.get(env);x=T.L.packet(env,obs,0,28000,np.zeros(2,np.float32),0);err=float(np.abs(H.rebuild(x,env._pose()[:3],meta[0],j['active_goal_xy'],meta[1])-x).max());assert err<=2e-6
                geometry.append(dict(id=goal['id'],physics_steps=0,valid=True,initial120_sha256=S.digest(x),goal_xy=env.goal_xy.tolist(),pose=list(env._pose()[:3]),identity_error=err,integration_sha256=R.arrays_hash(R.integration(env))))
                C.save(OUT/'geometry.json',geometry)
            finally:
                if env is not None:witness.events.pop(id(env),None);env.close()
        assert len(geometry)==16
    except BaseException:
        C.save(OUT/'preflight_failure.json',dict(traceback=traceback.format_exc(),physics_steps=0,optimizer_calls=0,geometry_commands_completed=len(geometry),Actor_updates=0));raise
    finally:
        if witness is not None:witness.close()
        if log is not None:log.close()
        if lock is not None:lock.close()
    assert sum(v.numel()*v.element_size() for v in model.state_dict().values())*3+32768<=p['model_bytes']
    # B's full-state-chain bound is229504B; compact worst-case raw+ZIP headers.
    assert 280*(1+3+24+8+16+8)+480+24+4096<=p['compact_bytes']
    reserve=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'];assert reserve<=p['global_bytes'];verify(p)
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,physics_steps=0,geometry_commands=16,optimizer_calls=0,Actor_updates=0,H3_input_and_single_blind_invariance=True,DQC_teacher_query_stopgradient=True,
        full_model_Adam_uncompressed_bound=sum(v.numel()*v.element_size() for v in model.state_dict().values())*3+32768,full_worstcase_reserved_bytes=reserve,protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T309 preflight PASS:16native geometry initializations,0physics/Adam/Actor')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();deadline=start+p['wall_seconds']
    native_deadline=deadline-p['final_reserve_seconds'];counts=mp.get_context('spawn').Array('q',4);workers=[];pipes=[];meter=None
    raw=[];replays=[];inflight=[];calls=[];holder={};exports={};state='FAILED';last_status=0;phase='STARTED';report=None
    def status(name,force=False):
        nonlocal last_status,phase
        phase=name
        if not force and time.time()-last_status<10:return
        last_status=time.time();C.save(OUT/'status.json',dict(state=name,pid=os.getpid(),main_epoch=start,updated_epoch=last_status,
            absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,commands=0 if meter is None else meter.commands,
            physics_steps=sum(counts[:]),complete_unique_cases=sum(r['complete'] for r in raw),exact_replays=len(replays),
            inflight=[j['id'] for j in inflight],head_optimizer_calls=len(calls),Actor_updates=0))
    def guard():
        assert time.time()<deadline-15;meter.hold();status(phase)
    def receive(pipe):
        while not pipe.poll(.25):status('NATIVE_ACTIVE');assert time.time()<native_deadline+5
        r=pipe.recv();assert 'error' not in r,r;return r
    def store(item,replay_partial=False):
        r=item['row'];data=packed(item['arrays']) if r['arm']=='center' or replay_partial else compact(item['arrays']);buf=io.BytesIO();np.savez_compressed(buf,**data);b=buf.getvalue()
        assert len(b)<=(p['episode_bytes'] if r['arm']=='center' or replay_partial else p['compact_bytes']);meter.charge_payload(len(b));f=OUT/'episodes'/(r['id']+('_partial_replay' if replay_partial else '')+'.npz');assert not f.exists();f.write_bytes(b)
        r.update(path=str(f),sha256=C.sha(f),saved_keys=list(data),compact_control=r['arm']!='center' and not replay_partial);raw.append(r);C.save(OUT/'cases.json',raw)
    def event(name,step):
        assert len(calls)<4*p['head_steps'];calls.append(dict(head=name,step=step,invoked=True))
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(calls[-1])+'\n')
    def alarm(*_):raise TimeoutError('T309 absolute finite pilot deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(deadline-start)-5))
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        meter=Budget(ROOT,OUT,p,start_epoch=start);meter.commands=16;status('STARTED',True);ctx=mp.get_context('spawn')
        for i in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=worker,args=(b,i,p,counts));w.start();b.close();workers.append(w);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(r['ready'] and r['policy_hash']==p['policy_hash'] and r['CPU_affinity']==[i+1] for i,r in enumerate(ready))
        C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        by={};jobs=[dict(j,deadline_epoch=native_deadline) for j in p['jobs']]
        for replay in [False,True]:
            for offset in range(0,len(jobs),4):
                guard();meter.checkpoint();group=jobs[offset:offset+4];assert meter.commands+len(group)<=p['max_commands']
                assert sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);inflight=group
                for pipe,j in zip(pipes,group):pipe.send(j)
                items=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(counts[:]);inflight=[]
                for item in items:
                    r=item['row'];complete=r['complete'] and not r['error']
                    if not replay or not complete:store(item,replay)
                    if replay and complete:
                        original=by[r['id']];assert r['array_hashes']==original['array_hashes'] and r['integration_sha256']==original['integration_sha256']
                        old=control_arrays(original) if original.get('compact_control') else B.arrays(original['path']);fresh=compact(item['arrays']) if original.get('compact_control') else item['arrays'];assert all(np.array_equal(v,fresh[k]) for k,v in old.items())
                        r['original_id']=original['id'];replays.append(r);C.save(OUT/'replays.json',replays)
                    elif not replay:by[r['id']]=r
                    assert complete,'Partial/censored native evidence retained, no continuation'
                status('REPLAY' if replay else 'COLLECTION',True)
        for pipe in pipes:pipe.send(None)
        for w in workers:w.join(5);assert not w.is_alive()
        assert len(raw)==len(replays)==128 and sum(r['physics_steps'] for r in raw+replays)==sum(counts[:])
        status('FITTING_HEADS',True);models,opts,losses,initial=fit(p,raw,guard,event,holder)
        status('CONTROL_VALIDATION',True);report=evaluate(p,raw,models);C.save(OUT/'metrics.json',report)
        buf=io.BytesIO();np.savez_compressed(buf,losses=losses);blob=buf.getvalue();assert len(blob)<=p['evidence_bytes'];meter.charge_payload(len(blob));(OUT/'head_evidence.npz').write_bytes(blob)
        verify(p);state='COMPLETE'
    except BaseException:
        C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),phase=phase,inflight=[j['id'] for j in inflight],physics_steps=sum(counts[:]),commands=0 if meter is None else meter.commands,head_optimizer_calls=len(calls),Actor_updates=0))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except OSError:pass
        for w in workers:
            w.join(3)
            if w.is_alive():w.kill();w.join(3)
        for name,q in holder.get('models',{}).items():
            try:
                payload=dict(model=q.state_dict(),optimizer=holder['optimizers'][name].state_dict(),initial_hash=holder['initial'],steps_invoked=sum(c['head']==name for c in calls),partial=state!='COMPLETE')
                buf=io.BytesIO();torch.save(payload,buf);blob=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);assert len(blob)<=p['model_bytes'];meter.charge_payload(len(blob));f=OUT/(name+'_critic.pt.gz');f.write_bytes(blob)
                exports[name]=dict(path=str(f),sha256=C.sha(f),model_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(payload['optimizer']))
            except BaseException:
                state='FAILED';C.save(OUT/(name+'_export_failure.json'),dict(traceback=traceback.format_exc()))
        signal.alarm(0)
        if meter is not None:meter.close()
    s=dict(state=state,pid=os.getpid(),commands=0 if meter is None else meter.commands,physics_steps=sum(counts[:]),
        raw_record_physics=sum(r['physics_steps'] for r in raw+replays),unique_cases=len(raw),exact_replays=len(replays),
        head_optimizer_calls=len(calls),Actor_updates=0,models=exports,all_workers_stopped=all(not w.is_alive() for w in workers),
        inflight=[j['id'] for j in inflight],main_epoch=start,absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,elapsed_s=time.time()-start,
        own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),combined_value_control_gate=None if report is None else report['combined_gate'],
        six_metre_navigation_gate_evaluated=False,fresh16_executed=False,reload78_executed=False,seven_eight_entered=False,automatic_next_training=False)
    C.save(OUT/'summary.json',s);C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["preflight","main"]);args=parser.parse_args();globals()[args.mode]()
