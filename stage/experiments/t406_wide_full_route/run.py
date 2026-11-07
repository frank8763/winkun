"""Initial full baseline, two failure-targeted PPO rounds, final full gate."""
from t406_core import *
def main():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    L.install_stop_handlers();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);old,wide=supports(q,base,p,lambda:None);start=time.time();p=dict(p,native_stop_epoch=start+p['main_seconds']);learnend=start+p['learning_seconds'];counts=mp.get_context('spawn').RawArray('q',5)
    phase='SETUP';commands=0;last=0.;pool=None;rows=[];reps=[];rounds=[];calls=dict(live=0,shadow=0);result={};status='FAILED'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);outside=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file() and OUT not in f.parents)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,learning_deadline=learnend,native_deadline=p['native_stop_epoch']))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and commands<=p['max_commands'] and sum(counts)<=p['max_physics']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert outside+p['final_bytes']<=2**30;save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,rounds=len(rounds),updates=calls,native_deadline=p['native_stop_epoch']))
    def lg():guard();assert time.time()<learnend-5
    def evaluate(jobs,role,full=False,seeds=None,references=None,model_path=None):
        nonlocal commands,phase
        phase=role;out=[];model=L.export(q,opt,OUT/'sampling.pt.gz') if model_path is None else dict(path=str(model_path),sha256=sha(model_path))
        if model_path is not None:
            stored=torch.load(io.BytesIO(gzip.decompress(Path(model_path).read_bytes())),weights_only=False,map_location='cpu');assert stored['hashes']==L.statehash(q,opt) and stored['policy_hash']==q.phash()
        for offset in range(0,len(jobs),4):
            group=jobs[offset:offset+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in group)<=p['max_physics'];ids=list(range(commands+1,commands+len(group)+1));commands+=len(group);assert commands+read(OUT/'preflight.json')['zero_physics_commands']<=p['max_commands'];t=time.time()
            requests=[dict(model=model['path'],job=j,first=0,seed=None if seeds is None else seeds[offset+i]) for i,j in enumerate(group)];batch=pool.request(requests)
            for i,((a,r),j,cid) in enumerate(zip(batch,group,ids)):
                key=role+'_'+str(offset+i) if seeds is not None else role+'_'+j['id'];record=dict(r,role=key,job_id=j['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[offset+i],loaded_model_path=model['path'],loaded_model_sha256=model['sha256'])
                if role=='initial_wide5':record.update(historical_reconstruction=True,independent_new_navigation_sample=False)
                if references is None:
                    f=OUT/'episodes'/f'{len(rows):04}.npz';N.write_npz(f,N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if full else 5*1024);record.update(path=str(f),sha256=sha(f),full=full);rows.append(record);filename='cases.jsonl'
                else:
                    ref=references[offset+i]
                    for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert record[k]==ref[k],k
                    record.update(exact=True,independent_sample=False);reps.append(record);filename='replays.jsonl'
                with (OUT/filename).open('a') as f:f.write(json.dumps(record)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=key,physics=r['physics'],elapsed_s=time.time()-t,error=None,pid=r['pid']))+'\n')
                out.append((a,record))
        return out
    def accepted_ids(rs):return sorted(r['job_id'] for r in rs if r['success'])
    def event(rnd,which):
        def emit(where,k):
            if where=='AFTER':calls[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([rnd,which,where,k])+'\n')
        return emit
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T406 native deadline')));signal.alarm(p['main_seconds']-2)
    try:
        L.export(q,opt,OUT/'initial.pt.gz');pool=Pool(p,counts,guard);commands=4
        for i,r in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,error=None,pid=r['pid']))+'\n')
        controls=evaluate(p['wide5_jobs'],'initial_wide5');assert all(r['success'] for a,r in controls)
        for (a,r),f in zip(controls,p['wide_history_files']):
            historical=N.S.arrays(f)
            for k in N.R.NAMES:assert equal(a[k],historical[k]),k
            assert equal(a['final_native_integration'],historical['final_native_integration'])
        initial_dev=[r for a,r in evaluate(p['main_jobs'],'initial_main54')];initial_old=[r for a,r in evaluate(base['final_old_jobs'],'initial_old275')]
        result['initial']=dict(main54=tally(initial_dev),old275=tally(initial_old),wide5=tally([r for a,r in controls]))
        old20ids={j['id'] for j in base['old_jobs']};guardjobs=base['old_jobs'];assert len(guardjobs)==20
        oldsafe={r['job_id'] for r in initial_old if r['job_id'] in old20ids and r['success']};widesafe=set(accepted_ids([r for a,r in controls]));initial_failures=[r['job_id'] for r in initial_dev if not r['success']]
        targets=[j for j in p['main_jobs'] if j['id'] in initial_failures]
        if not targets:targets=[j for j in base['final_old_jobs'] if j['id'] in {r['job_id'] for r in initial_old if not r['success']}]
        result['training_target_order']=[j['id'] for j in targets]
        reason='TWO_FIXED_ROUNDS_COMPLETE'
        for rnd in range(p['rounds']):
            if not targets:reason='INITIAL_WIDE_AND_OLD_ALL_SAFE';break
            assert time.time()+180<learnend;job=targets[rnd%len(targets)];before=L.state(q,opt);seeds=[[p['seed'],rnd,k] for k in range(4)];values=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,references=[r for a,r in values]);batch=[a for a,r in values]
            shadow,so=learner(base,p);L.load(shadow,so,before);folder=OUT/'blocks'/str(rnd);folder.mkdir();save(folder/'before_hashes.json',L.statehash(q,opt))
            try:
                live=fit(q,opt,batch,p,old,wide,rnd,lg,event(rnd,'live'));other=fit(shadow,so,batch,p,old,wide,rnd,lg,event(rnd,'shadow'));assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);candidate=L.export(q,opt,folder/'proposed.pt.gz')
                checks=[r for a,r in evaluate(guardjobs,f'proposal_{rnd}_old20')];widechecks=[r for a,r in evaluate(p['wide5_jobs'],f'proposal_{rnd}_wide5')]
                safeold=set(accepted_ids(checks));safewide=set(accepted_ids(widechecks));accept=live['numeric_accept'] and oldsafe<=safeold and widesafe<=safewide
                if accept:oldsafe=safeold;widesafe=safewide
                else:L.load(q,opt,before)
            except BaseException:L.load(q,opt,before);save(folder/'interrupted_rollback.json',dict(exact=digest(L.state(q,opt))==digest(before)));raise
            rr=dict(round=rnd,job=job,sampling=tally([r for a,r in values]),proposal=live,proposed_model=candidate,old20=tally(checks),wide5=tally(widechecks),accepted=accept,old_success_ids=sorted(oldsafe),wide_success_ids=sorted(widesafe),retained_hashes=L.statehash(q,opt));rounds.append(rr);save(folder/'result.json',rr);save(OUT/'rounds.json',rounds)
            if len(rounds)>=2 and not any(x['accepted'] for x in rounds[-2:]):reason='TWO_CONSECUTIVE_REJECTIONS';break
        result['reason']=reason;result['model']=L.export(q,opt,OUT/'candidate.pt.gz');dev=[r for a,r in evaluate(p['main_jobs'],'final_main54')];oldall=[r for a,r in evaluate(base['final_old_jobs'],'final_old275')];result.update(main54=tally(dev),old275=tally(oldall),fresh16=None,reload78=None,stable_wide_six=False)
        if all(r['success'] for r in [*dev,*oldall]):
            admission=read(OUT/'fresh_admission.json');assert admission['state']=='PASS_NO_PRIOR_WIDE_VERSION_NAVIGATION'
            for f,h in admission['locked_ledger_SHA'].items():assert sha(f)==h
            fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
            if all(r['success'] for r in fresh):
                lookup={r['job_id']:r for r in [*dev,*oldall,*fresh]};rr=evaluate(p['reload_jobs'],'reload78',references=[lookup[j['id']] for j in p['reload_jobs']],model_path=result['model']['path']);result.update(reload78=dict(exact=len(rr),independent_sample=False,loaded_final_artifact_sha256=result['model']['sha256']),stable_wide_six=len(rr)==78)
        status='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=sum(counts),commands=commands,updates=calls))
    finally:
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();s=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=calls,rounds=rounds,result=result,all_workers_stopped=True,all_native_sessions_closed=True,narrow_scene_paused=True,seven_eight_entered=False);save(OUT/'summary.json',s);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=calls));N.manifest(OUT)
    assert status=='COMPLETE'
if __name__=='__main__':main()
