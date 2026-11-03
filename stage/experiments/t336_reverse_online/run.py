from t336_core import *
from t336_worker import worker
import multiprocessing as mp,signal

def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    a=Learner(p);initial=a.state();fixed=digest(a.actor.fixed.state_dict());d=prior(p,a.actor);old=anchors(p,a.actor)
    b=copy.deepcopy(a);r=optimize(p,a,d,d,old,p['seed']+99,5,True,lambda:None,lambda _r:None)
    rr=optimize(p,b,d,d,old,p['seed']+99,5,True,lambda:None,lambda _r:None)
    assert r==rr and a.hashes()==b.hashes() and a.calls==b.calls==dict(actor=5,critic=5)
    a.rollback((initial['actor'],initial['actor_optimizer']))
    assert digest(a.actor_state())==digest((initial['actor'],initial['actor_optimizer']))
    assert digest(a.actor.fixed.state_dict())==fixed
    # Exercise the actual native worker deserialization and canonical batch1.
    q=Actor(p);x=d['x'][:1].numpy();ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    os.environ['STAGE_PROCESS_CPU']='1';proc=ctx.Process(target=worker,args=(child,0,p,None,True));proc.start();child.close();os.environ['STAGE_PROCESS_CPU']='0'
    parent.send(dict(policy_np=K.S.state_np(q),x=x));assert parent.poll(90);ans=parent.recv();proc.join(10)
    assert not proc.is_alive() and proc.exitcode==0
    with torch.no_grad():out=q.step(torch.tensor(x))
    assert ans['policy_hash']==digest(q.state_dict()) and np.array_equal(ans['mode'],out[0].numpy()) and np.array_equal(ans['mean'],out[1].numpy())
    # True native factory sanity, no stepping; prefix equality is checked online
    # against all12 saved arrays before every learning suffix.
    folder=OUT/'preflight';folder.mkdir();s=witness=None
    spec=importlib.util.spec_from_file_location('stage336_zero_native',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M);course=read(p['course_protocol']);M.install(course)
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            native,create,witness,reward=M.setup(folder/'native')
            s=M.R.Session(course,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero physics')))
            assert s.steps==0 and s.env.max_steps==40000
            ref=arrays(p['reference']);assert np.array_equal(s.initial120,ref['states'][0])
            with torch.no_grad():u=q.step(torch.tensor(s.initial120[None]))[0];v=q.fixed.step(torch.tensor(s.initial120[None]))[0]
            assert float((u-v).abs().max())<=2e-7
        finally:
            if s is not None:s.close()
            if witness is not None:witness.close()
    buf=io.BytesIO();torch.save(a.state(),buf);model_bytes=len(gzip.compress(buf.getvalue(),mtime=0));assert model_bytes<p['model_bytes']
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,synthetic_discarded_Adam_calls=20,
        actual_live_shadow_SAC_and_Actor_Adam_rollback_exact=True,frozen_base_unchanged=True,
        actual_spawn_worker_strict_load_and_batch1_exact=True,native_wide80_initial120_exact=True,
        initial_mode_max_error=float((u-v).abs().max()),synthetic_model_bytes=model_bytes,
        scientific_control_not_validated=True,protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T336 preflight PASS:0physics,20syntheticdiscardedAdam,realworker/native80/exactshadow/rollback')

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=p['absolute_deadline_epoch'];train_end=p['absolute_training_deadline_epoch']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None
    records=[];replays=[];blocks=[];models={};arm_results={};learners={};phase='STARTING';active=[];last=0.;state='FAILED'
    current=None;shadow=None;raw_updates=dict(live_actor=0,live_critic=0,shadow_actor=0,shadow_critic=0)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,
            deadline_epoch=deadline,training_deadline_epoch=train_end,physics_steps=sum(physical[:]),
            commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),cases=len(records),replays=len(replays),
            blocks=len(blocks),inflight=active,updates=raw_updates))
    def guard():
        assert time.time()<train_end;meter.hold();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        result=pipe.recv();assert 'error' not in result,result;return result
    def jobs_run(jobs,actor,arm,role,save_full=False,replay=False,readback=None):
        nonlocal phase,active
        phase=arm+'_'+role;out=[]
        for off in range(0,len(jobs),4):
            group=[]
            for j in jobs[off:off+4]:
                j=dict(j,id=arm+'_'+role+'_'+j['id'],arm=arm,role='old' if role not in ['training','frontier','wide_center'] else role,
                    source_reset_id=j.get('source_reset_id') or j['id'],policy_np=K.S.state_np(actor),policy_hash=digest(actor.state_dict()),
                    deadline_epoch=min(train_end,deadline-30) if role=='training' else deadline-30,
                    aux_eta=0.,mode='deterministic',group=j.get('group','stage336'),readback_model=readback)
                group.append(j)
            references=None
            for isrep in ([False,True] if replay else [False]):
                meter.physics=sum(physical[:]);meter.checkpoint()
                assert time.time()<deadline-60 and meter.commands+len(group)+p['preflight_geometry_commands']<=p['max_commands']
                assert meter.physics+sum(40000 if j['role']!='old' else j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);active=[j['id'] for j in group];status(True)
                for k,(pipe,j) in enumerate(zip(pipes,group)):
                    if isrep:j=dict(j,replay_arrays=references[k]['arrays'])
                    pipe.send(j)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);active=[]
                for k,item in enumerate(batch):
                    row,a=item['row'],item['arrays'];row.update(arm=arm,role=role,job_id=jobs[off+k]['id'])
                    assert row['complete'] and not row.get('error'),'Incomplete case is not a navigation failure rate'
                    if isrep:
                        assert row['array_hashes']==references[k]['row']['array_hashes'] and row['integration_sha256']==references[k]['row']['integration_sha256']
                        assert all(np.array_equal(v,references[k]['arrays'][name]) for name,v in a.items())
                        r=dict(id=row['id'],physics_steps=row['physics_steps'],full_arrays_native_exact=True,not_independent_sample=True)
                        replays.append(r)
                        with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                    else:
                        data=packed(a) if save_full else {name:a[name] for name in ['actions','native_counts','end_flags']}
                        buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue()
                        assert len(blob)<= (p['training_episode_bytes'] if save_full else p['compact_bytes'])
                        assert save_full or sum(not r['full_arrays_saved'] for r in records)<p['max_compact_cases']
                        meter.charge_payload(len(blob));path=OUT/'episodes'/(row['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                        row.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=save_full)
                        records.append(row);out.append(item)
                        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                if not isrep:references=batch
                status(True)
        return out
    def safe(items):return all(i['row']['success'] and not i['row']['collision'] for i in items)
    def counts(items):return {k:sum(bool(i['row'][k]) for i in items) for k in ['success','collision','timeout']}
    def suffix_job(ident,prefix,seed,deterministic=False):return dict(id=ident,prefix=prefix,seed=seed,deterministic=deterministic)
    def update_block(learner,prior_data,online,old,arm,roundno,warmup=False):
        nonlocal shadow,raw_updates
        init=learner.state();shadow=Learner(p);shadow.load(init);steps=p['critic_warmup_steps'] if warmup else p['updates_per_round'];seed=p['seed']+10000+roundno
        part=dict(arm=arm,round=roundno,warmup=warmup,seed=seed,steps=steps,input_hashes=learner.hashes())
        traces={}
        for replica,obj in [('live',learner),('shadow',shadow)]:
            def event(row):
                status()
                with (OUT/'optimizer_progress.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,round=roundno,warmup=warmup,replica=replica,**row),separators=(',',':'))+'\n')
            before=obj.calls.copy()
            try:traces[replica]=optimize(p,obj,prior_data,online,old,seed,steps,not warmup,guard,event)
            finally:
                raw_updates[replica+'_actor']+=obj.calls['actor']-before['actor'];raw_updates[replica+'_critic']+=obj.calls['critic']-before['critic']
        assert traces['live']==traces['shadow'] and learner.hashes()==shadow.hashes()
        part.update(output_hashes=learner.hashes(),exact_shadow=True,trace_sha256=digest(traces['live']),last=traces['live'][-1]);shadow=None
        return part
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T336 native absolute07:15')))
    signal.alarm(max(1,int(deadline-time.time())-10))
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';meter=K.Budget(ROOT,OUT,p,start_epoch=start)
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        for arm in p['arms']:
            current=Learner(p);learners[arm]=current;prior_data=prior(p,current.actor);old=anchors(p,current.actor);online_rows=[];frontier=0;rejects=0;accepted=0;stopped='ROUND_BUDGET'
            initial=jobs_run(p['old_jobs'],current.actor,arm,'initial_old20');assert safe(initial),'New zero latent head initial old20 failed; no RL for this arm'
            for roundno in range(p['rounds']):
                if time.time()>train_end-600:stopped='TRAINING_DEADLINE';break
                prefix=0 if arm=='fixed' else p['frontiers'][frontier]
                train=[suffix_job('r%d_n%d'%(roundno,k),prefix,p['seed']+roundno*10+k) for k in range(4)]
                batch=jobs_run(train,current.actor,arm,'training',save_full=True,replay=True)
                for item in batch:online_rows.append(tensor_data(item['arrays'],current.actor))
                online=combine(online_rows)
                if roundno==0:
                    b=update_block(current,prior_data,online,old,arm,-1,True);b['accepted']=None;blocks.append(b)
                    with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(b,separators=(',',':'))+'\n')
                saved=current.actor_state();b=update_block(current,prior_data,online,old,arm,roundno)
                guards=jobs_run(p['old_jobs'],current.actor,arm,'r%d_old20'%roundno);b['accepted']=safe(guards);b['guard']=counts(guards)
                if b['accepted']:accepted+=1;rejects=0
                else:current.rollback(saved);rejects+=1
                b['retained_hashes']=current.hashes();blocks.append(b)
                with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(b,separators=(',',':'))+'\n')
                tests=jobs_run([suffix_job('r%d_frontier'%roundno,prefix,p['seed']+999,True),suffix_job('r%d_start'%roundno,0,p['seed']+999,True)],current.actor,arm,'frontier')
                last_two=safe(batch[-p['advance_last_safe']:]);advance=arm=='reverse' and b['accepted'] and last_two and tests[0]['row']['success']
                oldprefix=prefix
                if advance and frontier+1<len(p['frontiers']):frontier+=1
                entry=dict(arm=arm,round=roundno,prefix=oldprefix,training=counts(batch),old20=counts(guards),accepted=b['accepted'],
                    frontier=counts(tests[:1]),original_wide_center=counts(tests[1:]),next_prefix=0 if arm=='fixed' else p['frontiers'][frontier])
                with (OUT/'rounds.jsonl').open('a') as f:f.write(json.dumps(entry,separators=(',',':'))+'\n')
                if rejects>=p['consecutive_rejections_stop']:stopped='TWO_CONSECUTIVE_OLD20_REJECTIONS';break
            arm_results[arm]=dict(accepted_blocks=accepted,stop_reason=stopped,final_prefix=0 if arm=='fixed' else p['frontiers'][frontier])
            path=OUT/(arm+'_final.pt.gz');info=export(p,current,path);meter.charge_payload(info['bytes']);models[arm]=info;save(OUT/'models.json',models)
        # Both arms receive exactly the registered coarse development checks.
        for arm,current in learners.items():
            wide=jobs_run(p['course_dev_jobs'],current.actor,arm,'wide5')
            original=jobs_run(p['original_jobs'],current.actor,arm,'original2')
            arm_results[arm].update(wide5=counts(wide),original2=counts(original))
        chosen=max(learners,key=lambda a:(arm_results[a]['wide5']['success'],arm_results[a]['original2']['success'],a=='reverse'))
        current=learners[chosen];oldcases=jobs_run(p['final_old_jobs'],current.actor,chosen,'old275');six=jobs_run(p['six_jobs'],current.actor,chosen,'six54')
        full=dict(chosen=chosen,old275=counts(oldcases),six54=counts(six),fresh16=None,reload78=None,stable=False)
        if safe(oldcases) and safe(six):
            fresh=jobs_run(p['fresh_jobs'],current.actor,chosen,'fresh16');full['fresh16']=counts(fresh)
            if safe(fresh):
                mapping={i['row']['job_id']:i for i in [*oldcases,*six,*fresh]}
                reload=jobs_run(p['reload_jobs'],current.actor,chosen,'reload78',readback=models[chosen]['path'])
                for item in reload:
                    original=mapping[item['row']['job_id']]
                    assert item['row']['array_hashes']==original['row']['array_hashes'] and item['row']['integration_sha256']==original['row']['integration_sha256']
                full['reload78']=counts(reload);full['stable']=safe(reload)
        save(OUT/'final_gate.json',full);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,physics_steps=sum(physical[:]),updates=raw_updates))
        if current is not None:
            try:
                if arm not in models:
                    path=OUT/(arm+'_interrupted.pt.gz');models[arm]=export(p,current,path)
            except BaseException:save(OUT/'partial_model_failure.json',dict(error=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in workers:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
        if meter:meter.close()
    summary=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=deadline,
        commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),physics_steps=sum(physical[:]),
        cases=len(records),training_episodes=sum(r['role']=='training' for r in records),exact_replays=len(replays),
        updates=raw_updates,arms=arm_results,models=models,all_workers_stopped=all(not proc.is_alive() for proc in workers),
        gate=read(OUT/'final_gate.json') if (OUT/'final_gate.json').exists() else None,
        original_7m_8m_entered=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
