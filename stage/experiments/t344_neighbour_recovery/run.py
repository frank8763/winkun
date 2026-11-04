from t344_core import *
import multiprocessing as mp,signal

def main():
    p=read(HERE/'protocol.json');verify(p);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and K.C.sha(OUT/'geometry.json')==pre['geometry_sha256'];assert not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();l=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],l['deadline_epoch']);learn_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');physical=ctx.Array('q',4);workers=[];pipes=[];records=[];replays=[];blocks=[];models={};result={};state='FAILED';phase='STARTING';active=[];commands=5;last=0.;updates=dict(live=0,shadow=0);obj=Learner(p)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<5:return
        last=time.time();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(physical[:]),inflight=active,updates=updates,completed_blocks=len(blocks)))
    def budget():
        assert K.S.used(OUT)<=p['final_bytes'] and K.S.used(ROOT/'stage/runs')<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    def guard():assert time.time()<learn_end-p['optimizer_admission_seconds'];budget();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        return pipe.recv()
    def jobs(js,role,q=None,full=False,replay=False,model=None):
        nonlocal phase,active,commands
        phase=role;q=obj.actor if q is None else q;out=[]
        for off in range(0,len(js),4):
            group=js[off:off+4];reference=None
            for rep in ([False,True] if replay else [False]):
                assert time.time()+p['per_batch_admission_seconds']<deadline;budget()
                assert commands+len(group)<=p['max_commands'] and sum(physical[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                commands+=len(group);active=[j['id'] for j in group];status(True)
                for k,(pipe,j) in enumerate(zip(pipes,group)):
                    command=dict(j,id=role+'_'+j['id'],role=role,source_reset_id=j.get('source_reset_id',j['id']),aux_eta=0.,mode='deterministic',
                        deterministic=role!='training',deadline_epoch=(learn_end-15 if role=='training' else deadline-15),policy_np=K.S.state_np(q),policy_hash=digest(q.state_dict()),readback_model=model)
                    if rep:command['replay_arrays']=reference[k]['arrays']
                    pipe.send(command)
                batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
                for k,(j,item) in enumerate(zip(group,batch)):
                    assert 'error' not in item,item
                    r,a=item['row'],item['arrays'];r['job_id']=j.get('source_reset_id',j['id'])
                    if not r['complete'] or r.get('error'):
                        b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];path=OUT/('partial_'+r['id']+'.npz');path.write_bytes(blob)
                        r.update(path=str(path),sha256=K.C.sha(path))
                        with (OUT/'partial_cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                        raise AssertionError('Incomplete raw preserved; not navigation failure')
                    if rep:
                        ref=reference[k];assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                        assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][name]) for name,v in a.items())
                        row=dict(id=r['id'],job_id=r['job_id'],physics_steps=r['physics_steps'],full_arrays_and_integration_exact=True,independent_sample=False)
                        replays.append(row)
                        with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
                    else:
                        data=packed(a) if full else {name:a[name] for name in ['actions','native_counts','end_flags']}
                        b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();assert len(blob)<=p['full_bytes' if full else 'compact_bytes']
                        assert sum(v['full_arrays_saved']==full for v in records)<p['max_full_cases' if full else 'max_compact_cases']
                        path=OUT/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full)
                        records.append(r);out.append(item)
                        with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(r,separators=(',',':'))+'\n')
                if not rep:reference=batch
                status(True)
        return out
    def counts(items):return {k:sum(bool(i['row'][k]) for i in items) for k in ['success','collision','timeout']}
    def event(replica,row):
        if row['stage']=='AFTER_ADAM':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(replica=replica,**row),separators=(',',':'))+'\n')
        status()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T344 bounded native deadline')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        for k in range(4):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,physical));proc.start();b.close();workers.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['CPU_affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[w.pid for w in workers],ready=ready))
        supportactor=Learner(p).actor.eval().requires_grad_(False);v=torch.load(io.BytesIO(gzip.decompress(Path(p['supporting_model']).read_bytes())),weights_only=False,map_location='cpu')
        supportactor.load_state_dict(dict(supportactor.state_dict(),**v['state']['actor']),strict=True);assert digest(supportactor.state_dict())==p['supporting_policy_hash']
        source_job=dict(p['course_dev_jobs'][1],id='known_T342_wide_dev1',source_reset_id='wide_dev_1',seed=0)
        source=jobs([source_job],'support_regeneration',supportactor,full=True,replay=True)[0]
        ref=p['supporting_source_row'];assert source['row']['success'] and not source['row']['collision'] and source['row']['integration_sha256']==ref['integration_sha256'] and source['row']['physics_steps']==ref['physics_steps']
        common=source['row']['array_hashes'].keys() & ref['array_hashes'].keys();assert all(source['row']['array_hashes'][k]==ref['array_hashes'][k] for k in common)
        save(OUT/'support_contract.json',dict(historical_regeneration_not_new_independent=True,common_arrays_exact=sorted(common),source_id=source['row']['id'],historical_source_id=ref['id'],integration_exact=True,safe=True))
        initial=jobs(p['course_dev_jobs'],'initial_wide5');result['initial_wide5']=counts(initial)
        # Exact history control for the starting policy, not a new success claim.
        prior=[json.loads(l) for l in (ROOT/'stage/runs/t340_demo_supported_recovery/cases.jsonl').read_text().splitlines()]
        for item in initial:
            ref=next(r for r in prior if r['role']=='wide5' and r['job_id']==item['row']['job_id']);assert item['row']['integration_sha256']==ref['integration_sha256']
            common=item['row']['array_hashes'].keys()&ref['array_hashes'].keys();assert all(item['row']['array_hashes'][k]==ref['array_hashes'][k] for k in common)
        successes=[]
        for rnd,job_id in enumerate(p['training_job_ids']):
            assert time.time()+300<learn_end;guard();base=next(j for j in p['course_dev_jobs'] if j['id']==job_id)
            train=[dict(base,id='r%d_n%d'%(rnd,k),source_reset_id=job_id,seed=p['training_seed']+rnd*10+k) for k in range(4)]
            batch=jobs(train,'training',full=True,replay=True)
            successes.extend(i for i in batch if i['row']['success'] and not i['row']['collision'])
            phase='RECOVERY_UPDATE_R'+str(rnd);status(True);before=obj.state();shadow=Learner(p);shadow.load(before);trace={};before_calls=obj.calls
            try:
                for replica,value in [('live',obj),('shadow',shadow)]:
                    groups=targets(p,value.actor,source,successes);trace[replica]=T.supported_update(p,value,batch,groups,rnd,guard,lambda row:event(replica,row))
                assert trace['live']==trace['shadow'] and obj.hashes()==shadow.hashes()
            except BaseException:
                obj.load(before);save(OUT/'interrupted_block_rollback.json',dict(round=rnd,Actor_Adam_before_and_restored={k:digest(v) for k,v in before.items()},restored_hashes=obj.hashes(),newly_retained_updates=0));raise
            block=dict(round=rnd,training_family=job_id,training_ids=[i['row']['id'] for i in batch],success_ids=[i['row']['id'] for i in successes],input_hashes={k:digest(v) for k,v in before.items()},output_hashes=obj.hashes(),trace_sha256=digest(trace['live']),exact_shadow=True,retained=True,live_Adam=obj.calls-before_calls)
            blocks.append(block)
            with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(block,separators=(',',':'))+'\n')
            centers=jobs(p['course_dev_jobs'],'round%d_wide5'%rnd)
            with (OUT/'rounds.jsonl').open('a') as f:f.write(json.dumps(dict(round=rnd,family=job_id,training=counts(batch),deterministic_wide5=counts(centers)),separators=(',',':'))+'\n')
        models['final']=T.export(p,obj,OUT/'final.pt.gz');save(OUT/'models.json',models)
        old=jobs(p['old_jobs'],'final_old20');wide=jobs(p['course_dev_jobs'],'final_wide5');original=jobs(p['original_jobs'],'final_original2')
        mapping={i['row']['job_id']:i['row'] for i in [*old,*wide,*original]}
        reload=jobs([*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']],'model_readback27',model=models['final']['path'])
        for item in reload:
            ref=mapping[item['row']['job_id']];assert item['row']['array_hashes']==ref['array_hashes'] and item['row']['integration_sha256']==ref['integration_sha256']
        result.update(old20=counts(old),wide5=counts(wide),original2=counts(original),model_readback27_exact=True,old275=None,six54=None,fresh16=None,reload78=None,stable=False)
        save(OUT/'result.json',result);state='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,inflight=active,commands=commands,physics_steps=sum(physical[:]),updates=updates))
        if 'final' not in models:
            try:models['interrupted']=T.export(p,obj,OUT/'interrupted.pt.gz')
            except BaseException:save(OUT/'partial_model_error.json',dict(error=traceback.format_exc()))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in workers:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(physical[:]),cases=len(records),exact_replays=len(replays),new_training_collection=sum(r['role']=='training' for r in records),updates=updates,blocks=len(blocks),models=models,result=result,all_workers_stopped=all(not w.is_alive() for w in workers),stable_six=False,entered7or8=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    save(OUT/'summary.json',s);save(OUT/'status.json',s);print(json.dumps(s))

if __name__=='__main__':main()
