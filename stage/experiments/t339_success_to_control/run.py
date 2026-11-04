from t339_core import *
from t339_worker import worker
import multiprocessing as mp,signal

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();launch=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],launch['native_deadline_epoch']);train_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None
    records=[];replays=[];blocks=[];models={};phase='STARTING';active=[];last=0.;state='FAILED'
    current=Learner(p,'capg');raw_updates=dict(live_actor=0,shadow_actor=0);result={}
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<10:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,
            deadline_epoch=deadline,training_deadline_epoch=train_end,physics_steps=sum(physical[:]),
            commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),cases=len(records),replays=len(replays),blocks=len(blocks),inflight=active,updates=raw_updates))
    def guard():
        assert time.time()<train_end;meter.hold();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-15
        return pipe.recv()
    def jobs_run(jobs,actor,arm,role,save_full=False,replay=False,readback=None):
        nonlocal phase,active
        phase=arm+'_'+role;out=[]
        for off in range(0,len(jobs),4):
            group=[]
            for j in jobs[off:off+4]:
                j=dict(j,id=arm+'_'+role+'_'+j['id'],arm=arm,role='old' if role not in ['training','frontier','baseline_frontier','verification'] else role,
                    source_reset_id=j.get('source_reset_id') or j['id'],policy_np=K.S.state_np(actor),policy_hash=digest(actor.state_dict()),
                    deadline_epoch=min(train_end,deadline-30) if role=='training' else deadline-30,
                    aux_eta=0.,mode='deterministic',group=j.get('group','stage338'),readback_model=readback)
                group.append(j)
            references=None
            for isrep in ([False,True] if replay else [False]):
                meter.physics=sum(physical[:]);meter.checkpoint()
                assert time.time()<deadline-p['per_batch_admission_seconds'] and meter.commands+len(group)+p['preflight_geometry_commands']<=p['max_commands']
                assert meter.physics+sum(40000 if j['role']!='old' else j['case']['step_cap'] for j in group)<=p['max_physics']
                meter.commands+=len(group);active=[j['id'] for j in group];status(True)
                for k,(pipe,j) in enumerate(zip(pipes,group)):
                    if isrep:j=dict(j,replay_arrays=references[k]['arrays'])
                    if j.get('expected_path'):j=dict(j,replay_arrays=arrays(j['expected_path']))
                    pipe.send(j)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];meter.physics=sum(physical[:]);active=[]
                for k,item in enumerate(batch):
                    assert 'error' not in item,item
                    row,a=item['row'],item['arrays'];row.update(arm=arm,role=role,job_id=jobs[off+k]['id'])
                    if isrep:
                        assert row['array_hashes']==references[k]['row']['array_hashes'] and row['integration_sha256']==references[k]['row']['integration_sha256']
                        assert all(np.array_equal(v,references[k]['arrays'][name]) for name,v in a.items())
                        r=dict(id=row['id'],physics_steps=row['physics_steps'],full_arrays_native_exact=True,not_independent_sample=True)
                        replays.append(r)
                        with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                    else:
                        if group[k].get('expected_path'):
                            expected=arrays(group[k]['expected_path']);assert set(a)==set(expected) and all(np.array_equal(v,expected[name]) for name,v in a.items())
                            row['cached_replay_not_new_training']=True
                        data=packed(a) if save_full else {name:a[name] for name in ['actions','native_counts','end_flags']}
                        buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue()
                        assert len(blob)<= (p['training_episode_bytes'] if save_full else p['compact_bytes'])
                        assert save_full or sum(not r['full_arrays_saved'] for r in records)<p['max_compact_cases']
                        meter.charge_payload(len(blob));path=OUT/'episodes'/(row['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                        row.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=save_full)
                        records.append(row);out.append(item)
                        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                assert all(i['row']['complete'] and not i['row'].get('error') for i in batch),'Incomplete rows preserved; not navigation failures'
                if not isrep:references=batch
                status(True)
        return out
    def safe(items):return all(i['row']['success'] and not i['row']['collision'] for i in items)
    def counts(items):return {k:sum(bool(i['row'][k]) for i in items) for k in ['success','collision','timeout']}
    def suffix_job(ident,prefix,seed,deterministic=False):return dict(id=ident,prefix=prefix,seed=seed,deterministic=deterministic)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T3391800s native deadline')))
    signal.alarm(max(1,int(deadline-time.time())-10))
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';meter=K.Budget(ROOT,OUT,p,start_epoch=start)
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        verifyjobs=[dict(id='cached'+str(k),prefix=240,seed=r['seed'],deterministic=False,expected_path=r['path']) for k,r in enumerate(p['cached_training_rows'])]
        verified=jobs_run(verifyjobs,current.actor,'capg','verification')
        assert len(verified)==4 and all(v['row']['cached_replay_not_new_training'] for v in verified)
        result['cached_complete_native_exact']=True
        tests=[suffix_job('p'+str(k),k,p['seed']+999,True) for k in p['test_prefixes']]
        initial=jobs_run(tests,current.actor,'capg','baseline_frontier')
        result['before']=[dict(prefix=i['row']['prefix'],success=i['row']['success'],collision=i['row']['collision'],timeout=i['row']['timeout'],remaining_m=i['row']['remaining_m']) for i in initial]
        assert time.time()+p['optimizer_admission_seconds']<train_end
        phase='ONE_BATCH_PPO_UPDATE';initial_state=current.state();shadow=Learner(p,'capg');shadow.load(initial_state)
        traces={};block=dict(input_hashes=current.hashes(),epochs=p['epochs'],cached_ids=[r['id'] for r in p['cached_training_rows']])
        for replica,obj in [('live',current),('shadow',shadow)]:
            def event(row):
                status()
                with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(replica=replica,**row),separators=(',',':'))+'\n')
            oldcalls=obj.calls
            try:traces[replica]=optimize(p,obj,cached(p),guard,event)
            finally:raw_updates[replica+'_actor']+=obj.calls-oldcalls
        assert traces['live']==traces['shadow'] and current.hashes()==shadow.hashes()
        block.update(output_hashes=current.hashes(),trace_sha256=digest(traces['live']),exact_shadow=True)
        blocks.append(block);save(OUT/'block.json',block)
        models['capg']=export(p,current,OUT/'updated_final.pt.gz');meter.charge_payload(models['capg']['bytes']);save(OUT/'models.json',models)
        updated=jobs_run(tests,current.actor,'capg','frontier',save_full=True,replay=True,readback=models['capg']['path'])
        old=jobs_run(p['old_jobs'],current.actor,'capg','final_old20')
        result['after']=[dict(prefix=i['row']['prefix'],success=i['row']['success'],collision=i['row']['collision'],timeout=i['row']['timeout'],remaining_m=i['row']['remaining_m']) for i in updated]
        result['old20']=counts(old)
        result['learned48_diagnostic']=not result['before'][0]['success'] and result['after'][0]['success'] and not result['after'][0]['collision']
        save(OUT/'result.json',result);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,physics_steps=sum(physical[:]),updates=raw_updates))
        if 'capg' not in models:
            try:models['capg']=export(p,current,OUT/'interrupted.pt.gz')
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
        commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),physics_steps=sum(physical[:]),cases=len(records),exact_replays=len(replays),
        new_training_collection=0,cached_training_unique=4,cached_trajectories_native_reverified=result.get('cached_complete_native_exact',False),updates=raw_updates,
        result=result,models=models,all_workers_stopped=all(not proc.is_alive() for proc in workers),stable_six=False,
        full_old275_six54_fresh16_reload78='NOT_EXECUTED',original_7m_8m_entered=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
