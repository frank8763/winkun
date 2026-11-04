from t340_core import *
from t340_worker import worker
import multiprocessing as mp,signal

def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();launch=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],launch['native_deadline_epoch']);train_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];meter=None
    records=[];replays=[];blocks=[];models={};phase='STARTING';active=[];last=0.;state='FAILED'
    current=Learner(p,'capg');raw_updates=dict(live_warm=0,shadow_warm=0,live_actor=0,shadow_actor=0);result={};arm='capg'
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
    def update_event(kind,replica,row):
        status()
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,replica=replica,**row),separators=(',',':'))+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T340 bounded native deadline')))
    signal.alarm(max(1,int(deadline-time.time())-10))
    try:
        assert read(OUT/'preflight.json')['state']=='PASS';meter=K.Budget(ROOT,OUT,p,start_epoch=start)
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes]
        assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        phase='DEMO_LOCATION_WARMSTART';current.opt=torch.optim.Adam(current.actor.trainable(),lr=p['warm_lr'])
        initial_state=current.state();shadow=Learner(p,'capg');shadow.load(initial_state);traces={}
        for replica,obj in [('live',current),('shadow',shadow)]:
            groups=demos(p,obj.actor)
            def event(row):
                if row['stage']=='AFTER_ADAM':raw_updates[replica+'_warm']+=1
                update_event('warm',replica,row)
            traces[replica]=warmup(p,obj.actor,obj.opt,groups,guard,event)
        assert traces['live']==traces['shadow'] and current.hashes()==shadow.hashes()
        warm=dict(input_hashes={k:digest(v) for k,v in initial_state.items()},output_hashes=current.hashes(),trace_sha256=digest(traces['live']),steps=p['warm_steps'],exact_shadow=True)
        save(OUT/'warm_block.json',warm);models['warm']=export(p,current,OUT/'warm.pt.gz');meter.charge_payload(models['warm']['bytes'])
        current.opt=torch.optim.Adam(current.actor.trainable(),lr=p['actor_lr'])
        tests=[suffix_job('p'+str(k),k,p['seed']+999,True) for k in p['frontiers']]
        warmtests=jobs_run(tests,current.actor,'capg','baseline_frontier');result['warm']=[i['row'] for i in warmtests]
        cursor=2 if warmtests[2]['row']['success'] else 1 if warmtests[1]['row']['success'] else 0
        successes=[];completed=0
        for rnd in range(p['rounds']):
            if time.time()+300>train_end:result['stop_reason']='TRAINING_DEADLINE_RESERVE';break
            prefix=p['frontiers'][cursor]
            train=[suffix_job('r%d_n%d'%(rnd,k),prefix,p['seed']+10000+rnd*10+k) for k in range(4)]
            batch=jobs_run(train,current.actor,'capg','training',save_full=True,replay=True)
            for item in batch:
                if item['row']['success'] and not item['row']['collision']:successes.append(item)
            init=current.state();shadow=Learner(p,'capg');shadow.load(init);traces={}
            part=dict(round=rnd,prefix=prefix,input_hashes=current.hashes(),training_ids=[i['row']['id'] for i in batch],success_ids=[i['row']['id'] for i in successes],epochs=p['epochs'])
            for replica,obj in [('live',current),('shadow',shadow)]:
                group=demos(p,obj.actor,successes);before=obj.calls
                try:traces[replica]=supported_update(p,obj,batch,group,rnd,guard,lambda row:update_event('r'+str(rnd),replica,row))
                finally:raw_updates[replica+'_actor']+=obj.calls-before
            assert traces['live']==traces['shadow'] and current.hashes()==shadow.hashes()
            part.update(output_hashes=current.hashes(),trace_sha256=digest(traces['live']),exact_shadow=True,retained=True)
            blocks.append(part)
            with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(part,separators=(',',':'))+'\n')
            check=jobs_run([dict(j,id='r'+str(rnd)+'_'+j['id']) for j in tests],current.actor,'capg','frontier')
            advance=counts(batch)['success']>=2 and check[cursor]['row']['success'] and not check[cursor]['row']['collision']
            nextcursor=min(cursor+1,len(p['frontiers'])-1) if advance else cursor
            entry=dict(round=rnd,prefix=prefix,training=counts(batch),centers=[dict(prefix=i['row']['prefix'],success=i['row']['success'],collision=i['row']['collision'],timeout=i['row']['timeout']) for i in check],next_prefix=p['frontiers'][nextcursor],demo_success_groups=len(successes))
            with (OUT/'rounds.jsonl').open('a') as f:f.write(json.dumps(entry,separators=(',',':'))+'\n')
            cursor=nextcursor;completed+=1
        models['final']=export(p,current,OUT/'final.pt.gz');meter.charge_payload(models['final']['bytes']);save(OUT/'models.json',models)
        old=jobs_run(p['old_jobs'],current.actor,'capg','final_old20')
        wide=jobs_run(p['course_dev_jobs'],current.actor,'capg','wide5')
        original=jobs_run(p['original_jobs'],current.actor,'capg','original2')
        result.update(completed_rounds=completed,final_prefix=p['frontiers'][cursor],old20=counts(old),wide5=counts(wide),original2=counts(original),old275=None,six54=None,fresh16=None,reload78=None,stable=False)
        if safe(wide) and safe(original):
            oldall=jobs_run(p['final_old_jobs'],current.actor,'capg','old275');six=jobs_run(p['six_jobs'],current.actor,'capg','six54')
            result.update(old275=counts(oldall),six54=counts(six))
            if safe(oldall) and safe(six):
                fresh=jobs_run(p['fresh_jobs'],current.actor,'capg','fresh16');result['fresh16']=counts(fresh)
                if safe(fresh):
                    mapping={i['row']['job_id']:i for i in [*oldall,*six,*fresh]}
                    reload=jobs_run(p['reload_jobs'],current.actor,'capg','reload78',readback=models['final']['path'])
                    for item in reload:
                        source=mapping[item['row']['job_id']]
                        assert item['row']['array_hashes']==source['row']['array_hashes'] and item['row']['integration_sha256']==source['row']['integration_sha256']
                    result['reload78']=counts(reload);result['stable']=safe(reload)
        save(OUT/'result.json',result);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,physics_steps=sum(physical[:]),updates=raw_updates))
        if 'final' not in models:
            try:models['final']=export(p,current,OUT/'interrupted.pt.gz')
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
        new_training_collection=sum(v['role']=='training' for v in records),updates=raw_updates,result=result,models=models,all_workers_stopped=all(not proc.is_alive() for proc in workers),
        stable_six=result.get('stable',False),original_7m_8m_entered=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
