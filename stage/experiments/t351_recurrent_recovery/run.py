from t351_core import *
import multiprocessing as mp, signal


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
    launch=read(OUT/'launch.json');deadline=min(start+p['wall_seconds'],launch['native_deadline_epoch']);learn_end=deadline-p['final_reserve_seconds']
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2);processes=[];pipes=[];rows=[];replays=[];blocks=[];commands=7
    obj=Learner(p);result={};phase='STARTING';active=[];updates=dict(live=0,shadow=0);state='FAILED';last=0.;model=None
    meter=C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,main_epoch=start,
            deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    def guard():assert time.time()<learn_end;meter.check();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        return pipe.recv()
    def short(r):
        rr=copy.deepcopy(r);hashes=rr.pop('array_hashes');rr['all_arrays_digest']=digest(hashes)
        if rr.get('full_arrays_saved'):rr['full_array_hashes']=hashes
        return rr
    def jobs(js,role,reference=None,readback=False):
        nonlocal phase,commands,active
        phase=role;out=[]
        for off in range(0,len(js),2):
            group=js[off:off+2];assert time.time()+p['per_batch_admission_seconds']<deadline
            assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            meter.check();commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):
                pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',learned=obj.actor.learned(),
                    policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=model['path'] if readback else None))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:
                    path=OUT/'episodes'/(r['id']+'_partial.npz');buff=io.BytesIO();np.savez_compressed(buff,**packed(a));blob=buff.getvalue();assert len(blob)<=p['partial_bytes'];path.write_bytes(blob)
                    with (OUT/'partial_cases.jsonl').open('a') as f:f.write(json.dumps(short(r),separators=(',',':'))+'\n')
                    raise RuntimeError('Partial native evidence preserved; no replacement job')
                if reference is not None:
                    ref=reference[r['job_id']];assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                    assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][name]) for name,v in a.items())
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(short(r))
                    with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(replays[-1],separators=(',',':'))+'\n')
                else:
                    oldfull=role=='initial_old275' and r['job_id'] in {j['id'] for j in p['old_jobs']}
                    newfull=(role=='initial_wide5' and r['job_id'] in ['wide_dev_2','wide_dev_4']) or role.startswith('train_')
                    full=oldfull or newfull;data=packed(a) if full else {name:a[name] for name in ['actions','native_counts','end_flags']}
                    buff=io.BytesIO();np.savez_compressed(buff,**data);blob=buff.getvalue();limit=p['old_full_bytes'] if oldfull else p['full_bytes'] if full else p['compact_bytes']
                    assert len(blob)<=limit,(r['id'],len(blob),limit)
                    path=OUT/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                    r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full,old_reference_full=oldfull)
                    saved=short(r);saved['saved_array_hashes']={name:digest(v) for name,v in data.items()};rows.append(saved)
                    assert sum(not x['full_arrays_saved'] for x in rows)<=p['max_compact_cases']
                    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(saved,separators=(',',':'))+'\n')
                out.append(item)
            status(True)
        return out
    def success_ids(items):return sorted(i['row']['job_id'] for i in items if i['row']['success'])
    def count(items):return {k:sum(bool(v['row'][k]) for v in items) for k in ['success','collision','timeout']}
    def evaluate(role):
        items=jobs([*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']],role)
        return items,dict(old20=count(items[:20]),wide5=count(items[20:25]),original2=count(items[25:]),
            old_success_ids=success_ids(items[:20]),wide_success_ids=success_ids(items[20:25]))
    def event(rnd,replica,row):
        if row['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,row['phase'],row['step']],separators=(',',':'))+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T351 native deadline')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[q.pid for q in processes],ready=ready))
        initial_old=jobs(p['final_old_jobs'],'initial_old275');byid={i['row']['job_id']:i for i in initial_old}
        ownold=[byid[j['id']] for j in p['old_jobs']];initial_wide=jobs(p['course_dev_jobs'],'initial_wide5');initial_narrow=jobs(p['original_jobs'],'initial_original2')
        supports=[i for i in initial_wide if i['row']['job_id'] in ['wide_dev_2','wide_dev_4']]
        historical=[json.loads(v) for v in Path(p['historical_cases']).read_text().splitlines()]
        for item in supports:
            old=next(r for r in historical if r['arm']=='memory' and r['role']=='final27' and r['job_id']==item['row']['job_id'])
            assert item['row']['success'] and item['row']['integration_sha256']==old['integration_sha256']
            assert all(item['row']['array_hashes'][k]==v for k,v in old['array_hashes'].items())
        jobs([j for j in p['course_dev_jobs'] if j['id'] in ['wide_dev_2','wide_dev_4']],'initial_support_replay',{i['row']['job_id']:i for i in supports})
        assert count(ownold)==dict(success=9,collision=11,timeout=0) and count(initial_wide)==dict(success=2,collision=2,timeout=1)
        result['initial']=dict(old275=count(initial_old),old20=count(ownold),wide5=count(initial_wide),original2=count(initial_narrow))
        retained_old=set(success_ids(ownold));retained_wide=set(success_ids(initial_wide));refs=reference_data(p,obj.actor,ownold,supports,guard);rejects=0
        for rnd in range(p['rounds']):
            course=p['course_dev_jobs'][2 if rnd==0 else 4];train=[]
            for k in range(4):
                j=copy.deepcopy(course);j.update(id=f'r{rnd}_n{k}',seed=p['seed']+100*rnd+k,stochastic=True);train.append(j)
            items=jobs(train,f'train_r{rnd}');jobs(train,f'train_replay_r{rnd}',{i['row']['job_id']:i for i in items})
            phase=f'UPDATE_r{rnd}';status(True);before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=Learner(p);shadow.load(before)
            try:
                live=update(p,obj,items,refs,rnd,guard,lambda r:event(rnd,'live',r));other=update(p,shadow,items,refs,rnd,guard,lambda r:event(rnd,'shadow',r))
                assert live==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
                proposed=obj.hashes();checked,metrics=evaluate(f'candidate_r{rnd}')
                keep=retained_old<=set(metrics['old_success_ids']) and retained_wide<=set(metrics['wide_success_ids'])
                if keep:retained_old=set(metrics['old_success_ids']);retained_wide=set(metrics['wide_success_ids']);rejects=0
                else:obj.load(before);rejects+=1
            except BaseException:
                obj.load(before);save(OUT/f'interrupted_r{rnd}.json',dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
            block=dict(round=rnd,input_hashes=input_hashes,proposed_hashes=proposed,output_hashes=obj.hashes(),update=live,retained=keep,
                metrics=metrics,live_Adam=p['epochs'],shadow_Adam=p['epochs'],exact_shadow=True)
            blocks.append(block)
            with (OUT/'blocks.jsonl').open('a') as f:f.write(json.dumps(block,separators=(',',':'))+'\n')
            if rejects>=2:break
        model=export(p,obj,OUT/'final_checkpoint.pt.gz');final,metrics=evaluate('final27');mapping={i['row']['job_id']:i for i in final}
        jobs([*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']],'model_readback27',mapping,True)
        result['final']=metrics;result.update(old275=None,six54=None,fresh16=None,reload78=None,stable_six=False)
        if metrics['old20']['success']==20 and metrics['wide5']['success']==5 and metrics['original2']['success']==2:
            old=jobs(p['final_old_jobs'],'final_old275');six=jobs(p['six_jobs'],'final_six54');result.update(old275=count(old),six54=count(six))
            if result['old275']['success']==275 and result['six54']['success']==54:
                fresh=jobs(p['fresh_jobs'],'fresh16');result['fresh16']=count(fresh)
                if result['fresh16']['success']==16:
                    ref={i['row']['job_id']:i for i in [*old,*six,*fresh]};rr=jobs(p['reload_jobs'],'release_reload78',ref,True)
                    result.update(reload78=dict(exact=len(rr)==78),stable_six=len(rr)==78)
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    if model is None:model=export(p,obj,OUT/'failed_current_checkpoint.pt.gz')
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,
        models=model,result=result,primary_cases=len(rows),replays=len(replays),blocks=len(blocks),all_workers_stopped=all(not q.is_alive() for q in processes),
        stable_six=bool(result.get('stable_six',False)),entered7or8=False,overnight_started=False,
        new_training_native_collection=sum(1 for r in rows if r['role'].startswith('train_')),quota_checks=meter.stats())
    save(OUT/'summary.json',s);save(OUT/'status.json',s);print(json.dumps(s),flush=True)


if __name__=='__main__':main()
