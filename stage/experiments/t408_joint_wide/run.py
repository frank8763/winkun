"""Joint compatibility before expensive full regression; LEFT then RIGHT PPO."""
from t408_core import *
def main():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();L.install_stop_handlers()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);broad=N.old_data(q,base,lambda:None);start=time.time();p=dict(p,native_stop_epoch=start+p['main_seconds']);learnend=start+p['learning_seconds'];counts=mp.get_context('spawn').RawArray('q',5)
    phase='SETUP';commands=0;last=0.;pool=None;rows=[];reps=[];rounds=[];calls=dict(live=0,shadow=0);result={};status='FAILED'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);outside=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file() and OUT not in f.parents)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,learning_deadline=learnend,native_deadline=p['native_stop_epoch']))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and commands+readiness_commands<=p['max_commands'] and sum(counts)<=p['max_physics']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert outside+p['final_bytes']<=2**30;save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,rounds=len(rounds),updates=calls,native_deadline=p['native_stop_epoch']))
    readiness_commands=read(OUT/'preflight.json')['zero_physics_commands']
    def lg():guard();assert time.time()<learnend-5
    def evaluate(jobs,role,full=False,seeds=None,references=None,model_path=None):
        nonlocal commands,phase
        phase=role;out=[];model=L.export(q,opt,OUT/'sampling.pt.gz') if model_path is None else dict(path=str(model_path),sha256=sha(model_path))
        if model_path is not None:
            z=torch.load(io.BytesIO(gzip.decompress(Path(model_path).read_bytes())),weights_only=False,map_location='cpu');assert z['hashes']==L.statehash(q,opt) and z['policy_hash']==q.phash()
        for off in range(0,len(jobs),4):
            js=jobs[off:off+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics'];ids=list(range(commands+1,commands+len(js)+1));commands+=len(js);assert commands+readiness_commands<=p['max_commands'];t=time.time()
            vals=pool.request([dict(model=model['path'],job=j,first=0,seed=None if seeds is None else seeds[off+i]) for i,j in enumerate(js)])
            for i,((a,r),j,cid) in enumerate(zip(vals,js,ids)):
                name=role+'_'+str(off+i) if seeds is not None else role+'_'+j['id'];row=dict(r,role=name,job_id=j['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[off+i],loaded_model_path=model['path'],loaded_model_sha256=model['sha256'])
                if references is None:
                    f=OUT/'episodes'/f'{len(rows):04}.npz';N.write_npz(f,N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if full else 5*1024);row.update(path=str(f),sha256=sha(f),full=full);rows.append(row);filename='cases.jsonl'
                else:
                    ref=references[off+i]
                    for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[k]==ref[k],k
                    row.update(exact=True,independent_sample=False);reps.append(row);filename='replays.jsonl'
                with (OUT/filename).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=name,physics=r['physics'],elapsed_s=time.time()-t,error=None,pid=r['pid']))+'\n')
                out.append((a,row))
        return out
    def joint(role):
        old=[r for a,r in evaluate(p['old8_jobs'],role+'_old8')];left=[r for a,r in evaluate(p['left5_jobs'],role+'_left5')];right=[r for a,r in evaluate([p['right_job']],role+'_right')];allrows=[*old,*left,*right]
        return dict(old8=tally(old),left5=tally(left),right1=tally(right),policy_hash=q.phash(),accepted=all(r['success'] for r in allrows),success_ids=sorted(r['job_id'] for r in allrows if r['success']))
    def event(tag,which):
        def emit(where,k):
            if where=='AFTER':calls[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([tag,which,where,k])+'\n')
        return emit
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T408 native deadline')));signal.alarm(p['main_seconds']-2)
    try:
        pool=Pool(p,counts,guard);commands=4
        for i,r in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,error=None,pid=r['pid']))+'\n')
        initial_old=evaluate(p['old8_jobs'],'initial_old8',full=True);assert all(r['success'] for a,r in initial_old)
        for a,r in initial_old:assert r['array_digest']==p['expected_old8_array_digests'][r['job_id']]
        right0=evaluate([p['right_job']],'initial_right');assert right0[0][1]['success'];rightref=N.S.arrays(p['right_history_file'])
        for k in [*N.R.NAMES,'hidden_before','last_hidden_after','final_native_integration']:assert equal(right0[0][0][k],rightref[k]),k
        left0=[r for a,r in evaluate(p['left5_jobs'],'initial_left5')];result['initial']=dict(old8=tally([r for a,r in initial_old]),left5=tally(left0),right1=tally([r for a,r in right0]),old275=None,main54=None)
        g=groups(q,[a for a,r in initial_old],p,broad);before=L.state(q,opt);folder=OUT/'blocks/joint';folder.mkdir();save(folder/'before_hashes.json',L.statehash(q,opt));shadow,so=learner(base,p);L.load(shadow,so,before)
        try:
            live=consolidate(q,opt,g,p,lg,event('joint','live'));other=consolidate(shadow,so,g,p,lg,event('joint','shadow'));assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);proposed=L.export(q,opt,folder/'proposed.pt.gz');checks=joint('joint_proposal');accepted=checks['accepted']
            if not accepted:L.load(q,opt,before)
        except BaseException:L.load(q,opt,before);save(folder/'interrupted_rollback.json',dict(exact=digest(L.state(q,opt))==digest(before)));raise
        warm=dict(fit=live,proposed_model=proposed,checks=checks,accepted=accepted,retained_hashes=L.statehash(q,opt));save(folder/'result.json',warm);result['joint_consolidation']=warm
        reason='JOINT_COMPATIBILITY_REJECTED'
        if accepted:
            opt=torch.optim.Adam(N.trainable(q),lr=p['lr']);reason='TWO_DIRECTION_ROUNDS_COMPLETE'
            for rnd,job in enumerate(p['rl_jobs']):
                assert time.time()+180<learnend;before=L.state(q,opt);seeds=[[p['seed'],rnd,k] for k in range(4)];vals=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,references=[r for a,r in vals]);batch=[a for a,r in vals];shadow,so=learner(base,p);L.load(shadow,so,before);folder=OUT/'blocks'/str(rnd);folder.mkdir();save(folder/'before_hashes.json',L.statehash(q,opt))
                try:
                    live=rl_fit(q,opt,batch,g,p,rnd,lg,event(rnd,'live'));other=rl_fit(shadow,so,batch,g,p,rnd,lg,event(rnd,'shadow'));assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);model=L.export(q,opt,folder/'proposed.pt.gz');check=joint(f'RL_{rnd}_proposal');keep=live['numeric_accept'] and check['accepted']
                    if not keep:L.load(q,opt,before)
                except BaseException:L.load(q,opt,before);save(folder/'interrupted_rollback.json',dict(exact=digest(L.state(q,opt))==digest(before)));raise
                row=dict(round=rnd,job=job,sampling=tally([r for a,r in vals]),proposal=live,proposed_model=model,checks=check,accepted=keep,retained_hashes=L.statehash(q,opt));rounds.append(row);save(folder/'result.json',row);save(OUT/'rounds.json',rounds)
                if len(rounds)==2 and not any(x['accepted'] for x in rounds):reason='TWO_RL_REJECTIONS_RETAIN_JOINT_MODEL';break
        result.update(reason=reason,model=L.export(q,opt,OUT/'candidate.pt.gz'),main54=None,old275=None,fresh16=None,reload78=None,stable_wide_six=False)
        centres=evaluate(p['centres'],'final_centres');result['main_centres']=tally([r for a,r in centres]);evaluate(p['centres'],'final_centres_exact',references=[r for a,r in centres],model_path=result['model']['path'])
        if accepted:
            result['final_joint']=joint('final_joint');assert result['final_joint']['accepted']
            dev=[r for a,r in evaluate(p['main_jobs'],'main54')];old=[r for a,r in evaluate(base['final_old_jobs'],'old275')];result.update(main54=tally(dev),old275=tally(old))
            if all(r['success'] for r in [*dev,*old]):
                admission=read(OUT/'fresh_admission.json');assert admission['state']=='PASS_WIDE_HOLDOUT_UNCONSUMED'
                for f,h in admission['locked_ledger_SHA'].items():assert sha(f)==h
                fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
                if all(r['success'] for r in fresh):
                    lookup={r['job_id']:r for r in [*dev,*old,*fresh]};reload=evaluate(p['reload_jobs'],'reload78',references=[lookup[j['id']] for j in p['reload_jobs']],model_path=result['model']['path']);result.update(reload78=dict(exact=len(reload),independent_sample=False),stable_wide_six=len(reload)==78)
        status='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=sum(counts),commands=commands,updates=calls))
    finally:
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=calls,rounds=rounds,result=result,all_workers_stopped=True,all_native_sessions_closed=True,narrow_paused=True,seven_eight_entered=False);save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=calls));N.manifest(OUT)
    assert status=='COMPLETE'
if __name__=='__main__':main()
