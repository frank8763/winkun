"""Finite current-history warm, conditional consolidation, reverse PPO course."""
from night import *

def main():
    N.offline();p=load_plan();base=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();native_end=min(start+p['main_seconds'],p['native_stop_epoch']);learnend=min(start+p['learning_seconds'],p['learning_stop_epoch']);p=dict(p,native_stop_epoch=native_end)
    phase='START';last=0.;completed_physics=commands=0;pool=None;cases=[];replays=[];updates=[];rounds=[];result={};status='FAILED';train_index=0;save_index=0;oldrefs=None
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if OUT not in f.parents and f.is_file())
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=native_end,learning_deadline=learnend))
    def physical():return completed_physics+(sum(pool.counts) if pool else 0)
    def guard():
        nonlocal last
        check_stop();assert time.time()<native_end-5 and physical()<=p['max_physics'] and commands<=p['max_commands']
        if time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes'])
            assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if OUT not in f.parents and f.is_file())==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physical(),commands=commands,rounds=len(rounds),native_deadline=native_end,learning_deadline=learnend))
    def learn_guard():guard();assert time.time()<learnend-5
    def close_pool():
        nonlocal pool,completed_physics
        if pool:
            pool.close();completed_physics+=sum(pool.counts);pool=None
    def open_pool(n,tag):
        nonlocal pool,commands
        guard();pool=Pool(p,n,guard,tag);commands+=n
    def req(j,model,first=0,seed=None,hold_limit=None):return dict(job=j,model=str(model),first=first,seed=seed,hold_limit=hold_limit)
    def record(a,r,role,first=0,seed=None,full=False,reference=None):
        nonlocal save_index
        rec=dict(r,role=role,first=first,seed=seed)
        if reference is None:
            f=OUT/'episodes'/f'{save_index:04}.npz';save_index+=1
            data=N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']}
            N.write_npz(f,data,p['full_bytes'] if full else 5*1024);rec.update(path=str(f),sha256=sha(f),full=full);cases.append(rec);file='cases.jsonl'
        else:
            assert r['array_digest']==reference['array_digest'] and r['integration_sha256']==reference['integration_sha256'];rec.update(exact=True,independent_sample=False);replays.append(rec);file='replays.jsonl'
        with (OUT/file).open('a') as stream:stream.write(json.dumps(rec,separators=(',',':'))+'\n')
        return rec
    def batch(requests,roles,full=False,references=None):
        nonlocal commands
        values=[]
        for begin in range(0,len(requests),pool.n):
            guard();part=requests[begin:begin+pool.n];commands+=len(part);got=pool.request(part)
            for k,(a,r) in enumerate(got):
                request=part[k];assert r['complete'] and not r['administrative_censor']
                row=record(a,r,roles[begin+k],request['first'],request['seed'],full,None if references is None else references[begin+k]);values.append((a,row))
        return values
    def evaluate(model,tag,keep53=False):
        nonlocal phase
        phase=tag;old=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
        requests=[req(j,model) for j in old]+[req(p['centre_left'],model,265)]
        vals=batch(requests,[tag+'_'+j['id'] for j in old]+[tag+'_mode53'],full=False)
        oldpass=all(row['success'] for _,row in vals[:8]);mode=vals[-1][1]
        return dict(old8=dict(valid=8,success=sum(row['success'] for _,row in vals[:8]),contact=sum(row['collision'] for _,row in vals[:8]),timeout=sum(row['timeout'] for _,row in vals[:8])),mode53=mode,compatible=bool(oldpass and mode['success']))
    def event(stage,which):
        def emit(where,k):
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([stage,which,where,k])+'\n')
        return emit
    def supervised(stage,kind,initial,newrefs=None):
        nonlocal phase
        phase=stage;learn_guard();q,opt=new_learner(base,p,initial);sq,so=new_learner(base,p,initial);before=state(q,opt)
        if kind=='warm':
            live=L.warm_fit(q,opt,L.groups(q,p,[KNOWN_BRIDGE]),oldrefs,p,learn_guard,event(stage,'live'))
            other=L.warm_fit(sq,so,L.groups(sq,p,[KNOWN_BRIDGE]),oldrefs,p,learn_guard,event(stage,'shadow'))
        else:
            live=consolidate_fit(q,opt,oldrefs,newrefs,p,learn_guard,event(stage,'live'))
            other=consolidate_fit(sq,so,oldrefs,newrefs,p,learn_guard,event(stage,'shadow'))
        assert live==other and statehash(q,opt)==statehash(sq,so)
        folder=OUT/stage;folder.mkdir();model=export(q,opt,folder/'proposed.pt.gz');checks=evaluate(model['path'],stage+'_accept')
        if not checks['compatible']:load(q,opt,before);assert digest(state(q,opt))==digest(before)
        row=dict(stage=stage,kind=kind,initial_model=str(initial),proposal=live,model=model,checks=checks,accepted=checks['compatible'],retained_hashes=statehash(q,opt));updates.append(row);save(folder/'result.json',row);save(OUT/'supervised.json',updates)
        return model['path'] if row['accepted'] else None
    try:
        # Fresh engineering authorization: fixed equal total work, no training.
        perf=[];expected=None
        for n in [1,2,4]:
            phase='PARALLEL_PROBE_'+str(n);t=time.perf_counter();open_pool(n,'probe'+str(n));ready=time.perf_counter();digests=[]
            for begin in range(0,4,n):
                requests=[req(p['centres'][i%2],WARM_PARENT,hold_limit=50) for i in range(begin,min(4,begin+n))];commands+=len(requests)
                for i,(a,r) in enumerate(pool.request(requests)):
                    assert r['physics']==5000 and not r['complete'] and not a['end_flags'].any()
                    f=OUT/'probes'/f'{n}_{begin+i}.npz';N.write_npz(f,N.S.packed(a),64*1024);digests.append(digest(a))
            row=dict(concurrency=n,total_wall_s=time.perf_counter()-t,setup_wall_s=ready-t,after_ready_s=time.perf_counter()-ready,physics=20000,digests=digests)
            close_pool()
            if expected is None:expected=digests
            else:assert digests==expected,'Different trajectories across native process counts'
            perf.append(row);save(OUT/'parallel_probe.json',perf)
        n=min(perf,key=lambda r:r['total_wall_s'])['concurrency'];result['parallel_selected']=n;open_pool(n,'learning_workers')
        # Parent's own successful closed-loop old histories, not T255's visits.
        phase='COLLECT_T393_OWN_OLD8';oldjobs=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')];assert len(oldjobs)==8
        values=batch([req(j,SAFE_PARENT) for j in oldjobs],['own_old_'+j['id'] for j in oldjobs],full=True)
        assert all(r['success'] for _,r in values),'Parent own-old8 baseline differs from existing safe8 control'
        parent=fixed_actor(base,p,SAFE_PARENT).requires_grad_(False);oldrefs=targets(parent,[(a,0) for a,r in values]);save(OUT/'old_reference_contract.json',dict(source='T393 itself on actual old8 successful native histories',episodes=8,rows=sum(len(a['states']) for a,r in values),target_sha256=digest([(g['x'],g['target']) for g in oldrefs]),not_recovery_expert_for_off_trajectory=True))
        for (a,row),j in zip(values,oldjobs):audit_full(parent,a,0,None,p)
        selected=supervised('A_own_history_warm','warm',SAFE_PARENT)
        if selected is None:
            warm=fixed_actor(base,p,WARM_PARENT).requires_grad_(False)
            newdata=[(N.S.arrays(ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover53.npz'),265),(N.S.arrays(ROOT/'stage/runs/t399_rejected_warm_diagnostic/episodes/recover54.npz'),270)]
            selected=supervised('B_separate_consolidation','consolidation',WARM_PARENT,targets(warm,newdata))
        result['acquisition_and_old8_compatible']=selected is not None
        if selected is not None:
            q,opt=new_learner(base,p,selected);export(q,opt,OUT/'initial_RL.pt.gz');refs=L.groups(q,p,[KNOWN_BRIDGE]);level=0;rejects=stalled=0
            for rnd in range(p['rounds']):
                if time.time()+600>learnend:result['learning_stop_reason']='INSUFFICIENT_REGISTERED_ROUND_RESERVE';break
                first=p['prefix_holds'][level];phase='RL_ROUND_'+str(rnd);model=OUT/'sampling.pt.gz';export(q,opt,model)
                seeds=[[p['seed'],rnd,k] for k in range(4)];requests=[req(p['centre_left'],model,first,z) for z in seeds]
                vals=batch(requests,[f'train_{rnd}_{k}' for k in range(4)],full=True)
                again=batch(requests,[f'train_{rnd}_{k}_exact' for k in range(4)],references=[r for a,r in vals])
                items=[a for a,r in vals];before=state(q,opt);shadow,so=new_learner(base,p,selected);load(shadow,so,before);learn_guard()
                block=OUT/'blocks'/str(rnd);block.mkdir();save(block/'before_hashes.json',statehash(q,opt))
                try:
                    live=L.fit(q,opt,items,first,p,refs,oldrefs,learn_guard,event(rnd,'live'));other=L.fit(shadow,so,items,first,p,L.groups(shadow,p,[KNOWN_BRIDGE]),oldrefs,learn_guard,event(rnd,'shadow'))
                    assert live==other and statehash(q,opt)==statehash(shadow,so);proposal=export(q,opt,block/'proposed.pt.gz')
                    checks=evaluate(proposal['path'],'RL_'+str(rnd)) if live['numeric_accept'] else None
                    accepted=live['numeric_accept'] and checks is not None and checks['old8']['success']==8
                    if not accepted:load(q,opt,before);assert digest(state(q,opt))==digest(before)
                except BaseException:load(q,opt,before);save(block/'interrupted_rollback.json',dict(exact=digest(state(q,opt))==digest(before)));raise
                retained=export(q,opt,OUT/'retained.pt.gz');mode=batch([req(p['centre_left'],retained['path'],first)],[f'course_{rnd}'])[0][1]
                success=sum(row['success'] for a,row in vals);promote=accepted and success>=2 and mode['success'];rejects=0 if accepted else rejects+1;stalled=stalled+1 if success==0 and not mode['success'] else 0
                row=dict(round=rnd,prefix_holds=first,seeds=seeds,proposal=live,proposed_model=proposal,checks=checks,accepted=accepted,retained_hashes=statehash(q,opt),sampling_success=success,course_mode=mode,promote=promote,level_before=level)
                if promote and level<len(p['prefix_holds'])-1:level+=1
                row['level_after']=level;rounds.append(row);save(block/'result.json',row);save(OUT/'rounds.json',rounds)
                if rejects>=2 or stalled>=2:result['learning_stop_reason']='TWO_REJECTIONS_OR_TWO_NO_SUCCESS_FRONTIER_ROUNDS';break
            final=export(q,opt,OUT/'candidate.pt.gz');result['model']=final
        else:result['model']=dict(path=str(SAFE_PARENT),sha256=sha(SAFE_PARENT),policy_hash=parent.phash(),not_new_model=True);result['learning_stop_reason']='BOTH_PREDECLARED_COMPATIBILITY_MECHANISMS_REJECTED'
        phase='FINAL_CENTRES_AND_OLD8';model=result['model']['path'];result['final_old8_and_53']=evaluate(model,'final')
        vals=batch([req(j,model) for j in p['centres']],['final_main_'+j['id'] for j in p['centres']]);result['main_centres']=[row for a,row in vals]
        batch([req(j,model) for j in p['centres']],['final_main_readback_'+j['id'] for j in p['centres']],references=[row for a,row in vals])
        # Full acceptance needs an independently reserved full evidence cohort.
        # Never call a local centre/prefix result stable or consume fresh to select.
        result.update(old275=None,main54=None,fresh16=None,reload78=None,stable_six=False)
        if all(r['success'] for r in result['main_centres']) and result['final_old8_and_53']['old8']['success']==8:
            needed=p['full_gate_additional_bytes'];available=1024**3-N.K.S.used(ROOT/'stage/runs')
            result['full_gate_admission']=dict(required_bytes=needed,actual_available_bytes=available,ready=False,reason='Independent full78 evidence reservation required; not enough within locked own/global reserve' if available<needed else 'Full gate independent source not implemented in this night; no invented execution')
        result['seven_eight_not_started']='No complete six gate or registered7/8source'
        status='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=physical(),commands=commands))
    finally:
        close_pool();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=completed_physics,commands=commands,supervised=updates,rounds=rounds,result=result,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False,remote_pending=True)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=completed_physics,commands=commands));N.manifest(OUT)
    assert status=='COMPLETE'
if __name__=='__main__':main()
