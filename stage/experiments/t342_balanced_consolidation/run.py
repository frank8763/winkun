from t342_core import *
import multiprocessing as mp,signal

def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS';assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();launch=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],launch['deadline_epoch']);learn_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];records=[];phase='STARTING';active=[];commands=2;last=0.;updates=dict(live=0,shadow=0);models={};result={};state='FAILED';obj=Learner(p)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<5:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(physical[:]),inflight=active,updates=updates))
    def guard():
        assert time.time()<learn_end-p['optimizer_admission_seconds'];status()
        if time.time()-guard.last>1:
            assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes'];guard.last=time.time()
    guard.last=0.
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        return pipe.recv()
    def jobs(jobs,role,model=None):
        nonlocal phase,active,commands
        phase=role;out=[]
        for off in range(0,len(jobs),4):
            group=jobs[off:off+4];assert time.time()+p['per_batch_admission_seconds']<deadline
            assert commands+len(group)<=p['max_commands'] and sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):
                pipe.send(dict(j,id=role+'_'+j['id'],job_id=j['id'],role=role,source_reset_id=j.get('source_reset_id',j['id']),aux_eta=0.,mode='deterministic',seed=0,
                    deadline_epoch=deadline-15,policy_np=K.S.state_np(obj.actor),policy_hash=digest(obj.actor.state_dict()),readback_model=model))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for j,item in zip(group,batch):
                assert 'error' not in item,item
                r,a=item['row'],item['arrays'];r['job_id']=j['id']
                if not r['complete'] or r.get('error'):
                    b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes']
                    path=OUT/('partial_'+r['id']+'.npz');path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path))
                    with (OUT/'partial_cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                    raise AssertionError('Censored or invalid raw preserved, not navigation failure')
                b=io.BytesIO();np.savez_compressed(b,**{k:a[k] for k in ['actions','native_counts','end_flags']});blob=b.getvalue();assert len(blob)<=p['compact_bytes']
                assert len(records)<p['max_compact_cases'];path=OUT/'episodes'/(r['id']+'.npz');path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path));records.append(r);out.append(item)
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
            status(True)
        return out
    def counts(items):return {k:sum(bool(i['row'][k]) for i in items) for k in ['success','collision','timeout']}
    def safe(items):return all(i['row']['success'] and not i['row']['collision'] for i in items)
    def event(replica,row):
        if row['stage']=='AFTER_ADAM':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(replica=replica,**row),separators=(',',':'))+'\n')
        status()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T342 fixed native deadline')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        beforeold=jobs(p['old_jobs'],'initial_old20');beforewide=jobs(p['course_dev_jobs'][:1],'initial_wide_center')
        result['initial_old20']=counts(beforeold);result['initial_wide_center']=counts(beforewide)
        phase='FIXED_TARGET_DATA';status(True);data=dataset(p,obj.actor,guard);before=metrics(obj.actor,data,guard)
        initial=obj.state();shadow=Learner(p);shadow.load(initial);traces={};phase='BALANCED_FUNCTION_CONSOLIDATION';status(True)
        for name,v in [('live',obj),('shadow',shadow)]:traces[name]=fit(p,v,data,guard,lambda row:event(name,row))
        assert traces['live']==traces['shadow'] and obj.hashes()==shadow.hashes()
        after=metrics(obj.actor,data,guard);save(OUT/'fit.json',dict(before=before,after=after,exact_shadow=True,trace_sha256=digest(traces['live']),hashes=obj.hashes(),training_RL_updates=0))
        models['final']=export(p,obj,OUT/'final.pt.gz');save(OUT/'models.json',models)
        old=jobs(p['old_jobs'],'final_old20');wide=jobs(p['course_dev_jobs'],'final_wide5');original=jobs(p['original_jobs'],'final_original2')
        mapping={i['row']['job_id']:i for i in [*old,*wide,*original]}
        reload_jobs=[*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]
        repeated=jobs(reload_jobs,'model_readback27',models['final']['path'])
        for item in repeated:
            ref=mapping[item['row']['job_id']]['row'];assert item['row']['array_hashes']==ref['array_hashes'] and item['row']['integration_sha256']==ref['integration_sha256']
        result.update(old20=counts(old),wide5=counts(wide),original2=counts(original),model_readback27_exact=True,old275=None,six54=None,fresh16=None,reload78=None,stable=False)
        if safe(old) and safe(wide) and safe(original):
            oldall=jobs(p['final_old_jobs'],'old275');six=jobs(p['six_jobs'],'six54');result.update(old275=counts(oldall),six54=counts(six))
            if safe(oldall) and safe(six):
                fresh=jobs(p['fresh_jobs'],'fresh16');result['fresh16']=counts(fresh)
                if safe(fresh):
                    mapping={i['row']['job_id']:i for i in [*oldall,*six,*fresh]};loaded=jobs(p['reload_jobs'],'reload78',models['final']['path'])
                    for item in loaded:
                        ref=mapping[item['row']['job_id']]['row'];assert item['row']['array_hashes']==ref['array_hashes'] and item['row']['integration_sha256']==ref['integration_sha256']
                    result['reload78']=counts(loaded);result['stable']=safe(loaded)
        state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics_steps=sum(physical[:]),commands=commands,inflight=active,updates=updates))
        if 'final' not in models:
            try:models['interrupted']=export(p,obj,OUT/'interrupted.pt.gz')
            except BaseException:save(OUT/'partial_model_error.json',dict(error=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in workers:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    summary=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(physical[:]),updates=updates,RL_updates=0,
        result=result,models=models,all_workers_stopped=all(not w.is_alive() for w in workers),stable_six=result.get('stable',False),entered7or8=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',summary);save(OUT/'status.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
