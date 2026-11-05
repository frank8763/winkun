from t368_core import *


def main(outpath):
    out=Path(outpath);p=read(out/'protocol.json')
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
    assert not (out/'status.json').exists();deadline=min(start+p['wall_seconds'],read(out/'launch.json')['deadline_epoch']);learn_end=deadline-1200
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2);processes=[];pipes=[];rows=[];replays=[];blocks=[];commands=3
    phase='STARTING';last=0.;active=[];updates={'live':0,'shadow':0};result={};models={};objects={name:NightLearner(p,rho) for name,rho in p['arms'].items()};state='FAILED'
    meter=P.C.ReservedBudget(ROOT/'stage/runs',out,p['final_bytes'],1024**3,40*1024**3)
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();save(out/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,main_epoch=start,
            deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    def guard():assert time.time()<learn_end;meter.check();status()
    def receive(pipe):
        while not pipe.poll(.25):status();assert time.time()<deadline-10
        return pipe.recv()
    def compact(r,full=False):
        r=copy.deepcopy(r);hashes=r.pop('array_hashes');r['all_arrays_digest']=digest(hashes)
        if full:r['full_array_hashes']=hashes
        return r
    def jobs(js,role,obj,reference=None,model=None):
        nonlocal phase,commands,active
        phase=role;outitems=[]
        for offset in range(0,len(js),2):
            group=js[offset:offset+2];assert time.time()+60<deadline
            assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            meter.check();commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=obj.actor.rho,
                learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=model['path'] if model else None))
            batch=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:
                    b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['full_bytes']
                    path=out/'episodes'/(r['id']+'_partial.npz');path.write_bytes(blob)
                    with (out/'partial_cases.jsonl').open('a') as f:f.write(json.dumps(compact(r),separators=(',',':'))+'\n')
                    raise RuntimeError('Administrative partial preserved; no retry')
                if reference is not None:
                    ref=reference[r['job_id']]
                    assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                    assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);saved=compact(r);replays.append(saved)
                    with (out/'replays.jsonl').open('a') as f:f.write(json.dumps(saved,separators=(',',':'))+'\n')
                else:
                    full=role.startswith('train_');data=packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']}
                    b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();assert len(blob)<=p['full_bytes' if full else 'compact_bytes']
                    path=out/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                    r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full);saved=compact(r,full)
                    saved['saved_array_hashes']={k:digest(v) for k,v in data.items()};rows.append(saved)
                    assert sum(v['full_arrays_saved'] for v in rows)<=p['max_train_episodes']
                    assert sum(not v['full_arrays_saved'] for v in rows)<=p['max_compact_cases']
                    with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(saved,separators=(',',':'))+'\n')
                outitems.append(item)
            status(True)
        return outitems
    def count(items):return {k:sum(bool(v['row'][k]) for v in items) for k in ['success','collision','timeout']}
    def ids(items):return sorted(i['row']['job_id'] for i in items if i['row']['success'])
    def evaluate(role,obj):
        items=jobs([*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs'],*p['narrow_jobs']],role,obj)
        return items,dict(old20=count(items[:20]),wide5=count(items[20:25]),original2=count(items[25:27]),narrow0705=count(items[27:32]),narrow_success_ids=ids(items[27:32]),old_success_ids=ids(items[:20]),wide_success_ids=ids(items[20:25]))
    def event(arm,rnd,replica,r):
        if r['phase']=='AFTER':updates[replica]+=1
        with (out/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([arm,rnd,replica,r['phase'],r['step']],separators=(',',':'))+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded native phase')));signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(out/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        refs=references(p,next(iter(objects.values())).actor)
        for arm,obj in objects.items():
            initial,base=evaluate('initial_'+arm,obj)
            assert all(digest(it['row']['array_hashes'])==p['initial_expected_digests'][it['row']['job_id']] for it in initial), 'Current initial controls must exactly reproduce retained T365 final'
            ret_old=set(base['old_success_ids']);ret_wide=set(base['wide_success_ids']);ret_narrow=set(base['narrow_success_ids']);rejections=0;level=0
            for rnd in range(p['rounds']):
                if time.time()+180>=learn_end:break
                # Direct original-origin narrow experience; never a wide-stage replay.
                shift=p['training_shift_x_m']
                train=[];course=p['narrow_jobs'][0]
                for k in range(4):
                    j=copy.deepcopy(course);j.update(id=f'{arm}_r{rnd}_n{k}',stochastic=True,seed=p['seed']+100*rnd+k,circle_shift_x_m=shift);train.append(j)
                items=jobs(train,f'train_{arm}_r{rnd}',obj)
                jobs(train,f'replay_{arm}_r{rnd}',obj,{v['row']['job_id']:v for v in items})
                phase=f'UPDATE_{arm}_{rnd}';status(True);before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=NightLearner(p,obj.actor.rho);shadow.load(before)
                try:
                    live=optimize(p,obj,items,refs,rnd,guard,lambda r:event(arm,rnd,'live',r))
                    other=optimize(p,shadow,items,refs,rnd,guard,lambda r:event(arm,rnd,'shadow',r))
                    assert live==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
                    proposed=obj.hashes();checked,metrics=evaluate(f'candidate_{arm}_r{rnd}',obj)
                    center=[]
                    if p['curriculum']:
                        cj=copy.deepcopy(p['course_dev_jobs'][0]);cj.update(circle_shift_x_m=shift)
                        center=jobs([cj],f'course_center_{arm}_r{rnd}',obj)
                    keep=ret_old<=set(metrics['old_success_ids']) and ret_wide<=set(metrics['wide_success_ids']) and ret_narrow<=set(metrics['narrow_success_ids'])
                    if keep:
                        ret_old=set(metrics['old_success_ids']);ret_wide=set(metrics['wide_success_ids']);ret_narrow=set(metrics['narrow_success_ids']);rejections=0
                        if p['curriculum'] and count(items)['success']>=1 and center[0]['row']['success']:level=min(level+1,len(p['curriculum_shifts'])-1)
                    else:obj.load(before);rejections+=1
                except BaseException:
                    obj.load(before);save(out/f'interrupted_{arm}_{rnd}.json',dict(input_hashes=input_hashes,restored=obj.hashes(),retained_updates=0));raise
                block=dict(arm=arm,round=rnd,input_hashes=input_hashes,proposed_hashes=proposed,output_hashes=obj.hashes(),update=live,
                    kept=keep,metrics=metrics,train_count=count(items),circle_shift_x_m=shift,next_level=level,exact_shadow=True,
                    live_Adam=p['epochs'],shadow_Adam=p['epochs'],shadow_native_guard_reused_exact_same_policy_hash=True)
                blocks.append(block)
                with (out/'blocks.jsonl').open('a') as f:f.write(json.dumps(block,separators=(',',':'))+'\n')
                # Two rejected complete rounds give evidence and stop this arm;
                # the next preprogrammed mechanism is a separate protocol.
                if rejections>=2:break
            models[arm]=export(p,obj,out/(arm+'_final.pt.gz'));final,metrics=evaluate('final_'+arm,obj)
            jobs([*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs'],*p['narrow_jobs']],'model_readback_'+arm,obj,{v['row']['job_id']:v for v in final},models[arm])
            result[arm]=dict(initial=base,final=metrics,old275=None,six54=None,fresh16=None,reload78=None,stable_six=False,
                initial_old275_not_retested=True,old275_T351_reference=p['prior_initial_old275'] if p['initial_is_T349'] else None)
        # Selection uses consumed development performance, never fresh/test.
        # Primary wide successes, then old20, then preregistered arm order.
        selected=max(objects,key=lambda a:(result[a]['final']['wide5']['success'],result[a]['final']['old20']['success'],-list(objects).index(a)))
        obj=objects[selected];metrics=result[selected]['final']
        # Full release gates are explicitly outside this pilot's budget.
        state='COMPLETE'
    except BaseException:save(out/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError,BrokenPipeError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    for arm,obj in objects.items():
        if arm not in models:models[arm]=export(p,obj,out/(arm+'_partial_model.pt.gz'))
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,
        models=models,result=result,blocks=len(blocks),primary_cases=len(rows),replays=len(replays),selected=selected if state=='COMPLETE' else None,
        all_workers_stopped=all(not v.is_alive() for v in processes),stable_six=any(v.get('stable_six') for v in result.values()),entered7or8=False)
    save(out/'summary.json',s);save(out/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


if __name__=='__main__':main(sys.argv[1])
