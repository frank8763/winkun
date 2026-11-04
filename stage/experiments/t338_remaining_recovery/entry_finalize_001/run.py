from t338r_core import *
from t338r_worker import worker
import multiprocessing as mp,signal

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();launch=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],launch['native_deadline_epoch']);train_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None
    records=[];replays=[];blocks=[];models={};arm_results={};learners={};phase='STARTING';active=[];last=0.;state='FAILED';current=None;arm='tanh'
    raw_updates=dict(live_actor=0,shadow_actor=0)
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
        result=pipe.recv();return result
    def jobs_run(jobs,actor,arm,role,save_full=False,replay=False,readback=None):
        nonlocal phase,active
        phase=arm+'_'+role;out=[]
        for off in range(0,len(jobs),4):
            group=[]
            for j in jobs[off:off+4]:
                j=dict(j,id=arm+'_'+role+'_'+j['id'],arm=arm,role='old' if role not in ['training','frontier','baseline_frontier'] else role,
                    source_reset_id=j.get('source_reset_id') or j['id'],policy_np=K.S.state_np(actor),policy_hash=digest(actor.state_dict()),
                    deadline_epoch=min(train_end,deadline-30) if role=='training' else deadline-30,
                    aux_eta=0.,mode='deterministic',group=j.get('group','stage338'),readback_model=readback)
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
                    assert 'error' not in item,item
                    row,a=item['row'],item['arrays'];row.update(arm=arm,role=role,job_id=jobs[off+k]['id'])
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
                assert all(i['row']['complete'] and not i['row'].get('error') for i in batch),'Incomplete rows preserved; not navigation failures'
                if not isrep:references=batch
                status(True)
        return out
    def safe(items):return all(i['row']['success'] and not i['row']['collision'] for i in items)
    def counts(items):return {k:sum(bool(i['row'][k]) for i in items) for k in ['success','collision','timeout']}
    def suffix_job(ident,prefix,seed,deterministic=False):return dict(id=ident,prefix=prefix,seed=seed,deterministic=deterministic)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T338 absolute native deadline')))
    signal.alarm(max(1,int(deadline-time.time())-10))
    baseline=None
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';meter=K.Budget(ROOT,OUT,p,start_epoch=start)
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        baseline=read(OLD_OUT/'initial_baseline.json');save(OUT/'initial_baseline.json',dict(baseline,historical_reuse_not_new_native=True))
        for arm in p['arms']:
            current=Learner(p,arm);learners[arm]=current;completed=0;stopped='ROUND_BUDGET'
            for roundno in range(p['rounds']):
                if time.time()>train_end-180:stopped='TRAINING_DEADLINE';break
                if arm=='tanh' and roundno==0:
                    phase='cached_tanh_first_update';batch=cached(p);status(True)
                else:
                    batch=jobs_run([suffix_job('r%d_n%d'%(roundno,k),p['fixed_training_prefix'],p['seed']+roundno*10+k) for k in range(4)],current.actor,arm,'training',save_full=True,replay=True)
                initial_state=current.state();shadow=Learner(p,arm);shadow.load(initial_state);traces={}
                block=dict(arm=arm,round=roundno,input_hashes=current.hashes(),training_ids=[i['row']['id'] for i in batch],epochs=p['epochs'])
                for replica,obj in [('live',current),('shadow',shadow)]:
                    before=obj.calls
                    def event(r):
                        status()
                        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,round=roundno,replica=replica,**r),separators=(',',':'))+'\n')
                    try:traces[replica]=optimize(p,obj,batch,guard,event)
                    finally:raw_updates[replica+'_actor']+=obj.calls-before
                assert traces['live']==traces['shadow'] and current.hashes()==shadow.hashes()
                block.update(output_hashes=current.hashes(),trace_sha256=digest(traces['live']),exact_shadow=True,retained=True)
                blocks.append(block)
                with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(block,separators=(',',':'))+'\n')
                tests=jobs_run([suffix_job('r%d_p48'%roundno,240,p['seed']+999,True),suffix_job('r%d_p56'%roundno,280,p['seed']+999,True)],current.actor,arm,'frontier')
                row=dict(arm=arm,round=roundno,training=counts(batch),recovery48=counts(tests[:1]),control56=counts(tests[1:]),retained=True,not_full_six_navigation=True)
                with (OUT/'rounds.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                completed+=1
            arm_results[arm]=dict(completed_rounds=completed,stop_reason=stopped)
            models[arm]=export(p,current,OUT/(arm+'_final.pt.gz'));meter.charge_payload(models[arm]['bytes']);save(OUT/'models.json',models)
        for arm,current in learners.items():
            old=jobs_run(p['old_jobs'],current.actor,arm,'final_old20')
            wide=jobs_run(p['course_dev_jobs'],current.actor,arm,'wide5')
            original=jobs_run(p['original_jobs'],current.actor,arm,'original2')
            arm_results[arm].update(old20=counts(old),wide5=counts(wide),original2=counts(original))
        state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,physics_steps=sum(physical[:]),updates=raw_updates))
        if current is not None and arm not in models:
            try:models[arm]=export(p,current,OUT/(arm+'_interrupted.pt.gz'))
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
        commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),physics_steps=sum(physical[:]),cases=len(records),
        training_episodes=sum(r['role']=='training' for r in records),exact_replays=len(replays),updates=raw_updates,arms=arm_results,baseline=baseline,models=models,
        all_workers_stopped=all(not proc.is_alive() for proc in workers),full_old275_final_six54_fresh16_reload78='NOT_EXECUTED_BY_THIS_PILOT',
        stable_six=False,original_7m_8m_entered=False,cached_training_unique=4,cached_replays_reused=4,combined_commands=p['prior_commands']+p['preflight_geometry_commands']+(0 if meter is None else meter.commands),combined_physics=p['prior_physics']+sum(physical[:]),own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
