"""Exactly two adaptive correction cycles, then the complete conditional gate."""
from t404_core import *

def main():
    N.offline();p=plan();base=verify(p);N.configure(OUT);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q,opt=learner(base,p);old=old_groups(q,p);prior=N.S.arrays(REFERENCE)
    start=time.time();p=dict(p,native_stop_epoch=start+p['main_seconds'],learning_stop_epoch=start+p['learning_seconds']);counts=mp.get_context('spawn').RawArray('q',5)
    commands=0;phase='SETUP';last=0.;s=w=pool=None;cases=[];replays=[];cohorts=[];cycles=[];updates=[];calls=dict(live=0,shadow=0);result={};status='FAILED'
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file() and OUT not in f.parents)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=p['native_stop_epoch'],learning_deadline=p['learning_stop_epoch']))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and commands<=p['max_commands'] and sum(counts)<=p['max_physics']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert outside+p['final_bytes']+p['wide_reserved_bytes']<=2**30
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,cycles=len(cycles),updates=calls,native_deadline=p['native_stop_epoch']))
    def lg():guard();assert time.time()<p['learning_stop_epoch']-5
    def charge():counts[0]+=1;assert sum(counts)<=p['max_physics']
    def command(role,fn):
        nonlocal commands
        guard();commands+=1;before=counts[0];begin=time.time();error=None
        try:return fn()
        except BaseException:error=traceback.format_exc();raise
        finally:
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,role=role,physics=counts[0]-before,elapsed_s=time.time()-begin,error=error))+'\n')
    def record(a,r,role,full=False,reference=None):
        row=dict(r,role=role,array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),policy_hash=q.phash(),physics_steps=int(a['native_counts'].sum()))
        if reference is None:
            path=OUT/'episodes'/f'{len(cases):04}.npz';N.write_npz(path,N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if full else 5*1024)
            row.update(path=str(path),sha256=sha(path),full=full);cases.append(row);name='cases.jsonl'
        else:
            for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[k]==reference[k],k
            row.update(exact=True,independent_sample=False);replays.append(row);name='replays.jsonl'
        with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n')
        return row
    def construct(job):
        nonlocal s
        q.sample=False;q.reset_noise();m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
        s=L.Session(job,native,create,w,guard,charge,q);L.install_stop_handlers()
    def local_episode(job,role,first=0,reference=None,full=False):
        nonlocal phase,s
        phase=role
        def body():
            nonlocal s
            construct(job)
            for i in range(first):s.step(prior['actions'][i],owner=1);assert not s.done()
            while not s.done():s.step(owner=0)
            a=payload(s);r=record(a,s.result(),role,full,reference);s.close();s=None;return a,r
        return command(role,body)
    def evaluate(jobs,role,full=False,reference=None,first=0,seeds=None):
        nonlocal commands,phase
        phase=role;out=[];model=export(q,opt,OUT/'sampling.pt.gz')
        for begin in range(0,len(jobs),4):
            js=jobs[begin:begin+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics'];assert commands+len(js)<=p['max_commands']
            ids=list(range(commands+1,commands+len(js)+1));commands+=len(js);t=time.time()
            req=[dict(model=model['path'],job=j,first=first,seed=None if seeds is None else seeds[begin+i]) for i,j in enumerate(js)]
            values=pool.request(req)
            for i,((a,r),j,cid) in enumerate(zip(values,js,ids)):
                name=role+'_'+str(begin+i) if seeds is not None else role+'_'+j['id'];ref=None if reference is None else reference[begin+i]
                if reference is not None:q.load_learned(N.S.read_model(model['path']))
                rr=record(a,r,name,full,ref);rr['job_id']=j['id'];out.append((a,rr))
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=name,physics=r['physics'],elapsed_s=time.time()-t,error=None,pid=r['pid']))+'\n')
        return out
    def tally(rs):return dict(valid=len(rs),success=sum(r['success'] for r in rs),contact=sum(r['collision'] for r in rs),timeout=sum(r['timeout'] for r in rs),invalid=0,censored=0)
    oldjobs=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')];assert len(oldjobs)==8
    def compatibility(role):
        oldrows=[r for a,r in evaluate(oldjobs,role+'_old8')];_,mode=local_episode(p['centre_left'],role+'_mode53',265)
        return dict(old8=tally(oldrows),mode53=mode,accepted=all(r['success'] for r in oldrows) and mode['success'])
    def event(tag,which):
        def emit(where,k):
            if where=='AFTER':calls[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([tag,which,where,k])+'\n')
        return emit
    def update(kind,cycle,refs=None,batch=None,first=0):
        nonlocal opt
        tag=f'{cycle}_{kind}';folder=OUT/'blocks'/tag;folder.mkdir(parents=True);before=state(q,opt);shadow,so=learner(base,p);L.load(shadow,so,before)
        save(folder/'before_hashes.json',L.statehash(q,opt))
        try:
            if kind=='acquire':
                a=L.warm_fit(q,opt,correction_groups(q,cohorts,p),old,p,lg,event(tag,'live'),p['warm_steps']);b=L.warm_fit(shadow,so,correction_groups(shadow,cohorts,p),old,p,lg,event(tag,'shadow'),p['warm_steps'])
            elif kind=='consolidate':
                a=consolidate(q,opt,old,refs,p,lg,event(tag,'live'));b=consolidate(shadow,so,old,refs,p,lg,event(tag,'shadow'))
            else:
                a=L.fit(q,opt,batch,first,p,correction_groups(q,cohorts,p),old,lg,event(tag,'live'));b=L.fit(shadow,so,batch,first,p,correction_groups(shadow,cohorts,p),old,lg,event(tag,'shadow'))
            assert a==b and L.statehash(q,opt)==L.statehash(shadow,so);model=export(q,opt,folder/'proposed.pt.gz')
            row=dict(tag=tag,kind=kind,cycle=cycle,fit=a,proposed_model=model);updates.append(row);save(folder/'result.json',row);return row
        except BaseException:L.load(q,opt,before);save(folder/'rollback.json',dict(exact=digest(state(q,opt))==digest(before)));raise
    def teacher(cycle,source_array,context):
        nonlocal s,phase
        first=select_first(source_array,prior,p,context);folder=OUT/'teachers'/str(cycle);folder.mkdir(parents=True);data=search_buffer(first);cbegin=commands;pb=sum(counts)
        phase=f'TEACHER_{cycle}_CURRENT_PREFIX'
        def prefix():
            construct(p['centre_left'])
            if context=='goal':
                for i in range(first):s.step(prior['actions'][i],owner=1);assert not s.done()
            else:
                for i in range(first):
                    s.step(owner=0);assert not s.done();assert equal(s.rows[-1][1],source_array['actions'][i])
        command(phase,prefix);snapshot=s.capture();snapshot_hash=N.C.snapshot_hash(s);mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['teacher_sd']),(16,2)).copy();best=None;bestid=None;bestcost=float('inf');found=False;traces=[];queries=0
        for gen in range(2):
            ids=[]
            for j,theta in enumerate(teacher_candidates(mean,sd,best,cycle,gen,p)):
                lg();idx=12*gen+j;u=teacher_controls(theta,prior['actions'],first,p);data['coefficients'][idx]=theta;data['controls'][idx]=u
                for rep in range(2):
                    phase=f'TEACHER_{cycle}_QUERY_{idx}_REP_{rep}'
                    def query():
                        s.restore(snapshot);q.sample=False
                        for k,action in enumerate(u):
                            row,hd=s.step(action,owner=1,actor_active=False)
                            if rep==0:data['next_poses'][idx,k]=row[5];data['counts'][idx,k]=row[2];data['flags'][idx,k]=row[6];data['hold_sha256'][idx,k]=hd
                            else:assert equal(data['next_poses'][idx,k],row[5]) and data['counts'][idx,k]==row[2] and equal(data['flags'][idx,k],np.asarray(row[6],np.uint8)) and equal(data['hold_sha256'][idx,k],hd)
                            if s.done():break
                        assert s.done();n=k+1;rem=float(s.env._goal_state()[1]);cost=objective(u,n,s.rows[-1][6],rem);integ=N.K.R.integration(s.env)
                        if rep==0:data['lengths'][idx]=n;data['costs'][idx]=cost;data['remaining_m'][idx]=rem;data['final_integration'][idx]=integ
                        else:assert data['lengths'][idx]==n and data['costs'][idx]==cost and equal(data['final_integration'][idx],integ)
                    command(phase,query)
                N.write_npz(folder/'search.npz',data,p['search_bytes']);ids.append(idx);queries+=1;score=float(data['costs'][idx])
                if bestid is None or (score,idx)<(bestcost,bestid):best=theta.copy();bestid=idx;bestcost=score
                found=bool(data['flags'][idx,int(data['lengths'][idx])-1,1]) and not bool(data['flags'][idx,:,0].any())
                if found:break
            elite=sorted(ids,key=lambda i:(float(data['costs'][i]),i))[:3]
            if not found:mean=.5*mean+.5*data['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['teacher_min_sd']),.5*sd+.5*data['coefficients'][elite].std(0))
            traces.append(dict(generation=gen,ids=ids,elite=elite,mean_after=mean.tolist(),sd_after=sd.tolist(),best_id=int(bestid),goal_found=found))
            if found:break
        c=dict(cycle=cycle,context=context,first=first,policy_hash=q.phash(),snapshot_sha256=snapshot_hash,queries=queries,query_exact_repeats=queries,full_teacher_safe=found,training_labels_allowed=found,traces=traces,best_id=int(bestid),search_hash=digest(data),short_bridge_GOAL_required=False)
        if found:
            saved=None
            for rep in range(2):
                phase=f'TEACHER_{cycle}_FULL_REP_{rep}'
                def full():
                    nonlocal saved
                    s.restore(snapshot);q.sample=False
                    for action in data['controls'][bestid]:
                        s.step(action,owner=1)
                        if s.done():break
                    assert s.done();a=payload(s);assert a['end_flags'][-1,1] and not a['end_flags'][:,0].any()
                    if rep==0:saved=a;N.write_npz(folder/'full.npz',N.S.packed(a),p['full_bytes'])
                    else:assert digest(a)==digest(saved)
                command(phase,full)
            c.update(full_path=str(folder/'full.npz'),full_sha256=sha(folder/'full.npz'),full_hash=digest(saved),goal_clock_s=float(saved['native_counts'].sum()*.002))
        c.update(commands=commands-cbegin,physics=sum(counts)-pb);s.close();s=None;save(folder/'result.json',c);cohorts.append(c);save(OUT/'teachers.json',cohorts);return c
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T404 native deadline')));signal.alarm(p['main_seconds']-2)
    try:
        m,native,create,w,_=command('native_setup',lambda:N.S.P.C.B.setup(OUT/'native',base));L.install_stop_handlers();original=copy.deepcopy(m.ACTIVE)
        pool=Pool(p,counts,guard);commands+=4
        for i,ready in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands-3+i,role='worker_setup',physics=0,error=None,pid=ready['pid']))+'\n')
        result['initial_compatibility']=compatibility('initial');assert result['initial_compatibility']['accepted']
        origin,rr=local_episode(p['centres'][0],'diagnostic_origin_left',full=True);local_episode(p['centres'][0],'diagnostic_origin_left_exact',reference=rr)
        assert rr['array_digest']==p['expected_origin_digest']
        mode52,mr=local_episode(p['centre_left'],'diagnostic_mode52',260,full=True);local_episode(p['centre_left'],'diagnostic_mode52_exact',260,reference=mr);assert mr['array_digest']==p['expected_mode52_digest']
        reason='TWO_CYCLES_COMPLETE'
        for cycle in range(2):
            assert time.time()+600<p['learning_stop_epoch'];before=state(q,opt);context='origin'
            if cycle:
                origin,row=local_episode(p['centres'][0],f'current_origin_{cycle}',full=True)
                if row['success']:context='goal';origin=mode52
            c=teacher(cycle,origin,context)
            if not c['training_labels_allowed']:reason='NO_VERIFIED_FULL_RECOVERY_IN_FIXED_QUERY_BUDGET';break
            opt=optimizer(q,p);acq=update('acquire',cycle);new=targets(q,[x for x in cohorts if x['training_labels_allowed']]);opt=optimizer(q,p);con=update('consolidate',cycle,new);test=compatibility(f'cycle_{cycle}_supervised')
            row=dict(cycle=cycle,first=c['first'],context=context,acquire=acq['tag'],consolidate=con['tag'],supervised_checks=test,supervised_accepted=test['accepted'],RL=None)
            if not test['accepted']:
                L.load(q,opt,before);row['retained_hashes']=L.statehash(q,opt);cycles.append(row);save(OUT/'cycles.json',cycles);reason='SUPERVISED_COMPATIBILITY_REJECTED';break
            opt=optimizer(q,p);rlbefore=state(q,opt);seeds=[[p['seed'],cycle,k] for k in range(4)];vals=evaluate([p['centre_left']]*4,f'train_{cycle}',full=True,first=c['first'],seeds=seeds)
            evaluate([p['centre_left']]*4,f'train_{cycle}_exact',reference=[r for a,r in vals],first=c['first'],seeds=seeds)
            batch=[a for a,r in vals];rl=update('RL',cycle,batch=batch,first=c['first']);checks=compatibility(f'cycle_{cycle}_RL') if rl['fit']['numeric_accept'] else None
            accepted=rl['fit']['numeric_accept'] and checks is not None and checks['accepted']
            if not accepted:L.load(q,opt,rlbefore)
            row.update(RL=rl['tag'],RL_checks=checks,RL_accepted=accepted,sampling=tally([r for a,r in vals]),retained_hashes=L.statehash(q,opt));cycles.append(row);save(OUT/'cycles.json',cycles)
        result['reason']=reason;result['model']=export(q,opt,OUT/'candidate.pt.gz');result['final_compatibility']=compatibility('final')
        centre_values=evaluate(p['centres'],'final_centres');centre_rows=[r for a,r in centre_values];evaluate(p['centres'],'final_centres_reload',reference=centre_rows);result['main_centres']=tally(centre_rows)
        result.update(old275=None,main54=None,fresh16=None,reload78=None,stable_six=False)
        if all(r['success'] for r in centre_rows):
            dev=[r for a,r in evaluate(p['main_jobs'],'main54')];oldall=[r for a,r in evaluate(base['final_old_jobs'],'old275')];result.update(main54=tally(dev),old275=tally(oldall))
            if all(r['success'] for r in [*dev,*oldall]):
                admission=read(OUT/'fresh_admission.json');assert admission['state']=='PASS_NO_PRIOR_NEW_VERSION_NAVIGATION'
                for f,h in admission['locked_ledger_SHA'].items():assert sha(f)==h
                fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
                if all(r['success'] for r in fresh):
                    lookup={r['job_id']:r for r in [*dev,*oldall,*fresh]};reload=[r for a,r in evaluate(p['reload_jobs'],'reload78',reference=[lookup[j['id']] for j in p['reload_jobs']])];result.update(reload78=dict(exact=len(reload),independent_sample=False),stable_six=len(reload)==78)
        result['next_action']='NARROW_PASSED_PREPARE_LONGER_PROTOCOL' if result['stable_six'] else 'PAUSE_NARROW_PROCEED_WIDE_ADMISSION';status='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=sum(counts),commands=commands,updates=calls))
        if s is not None and s.rows:
            try:N.write_npz(OUT/'partial.npz',N.S.packed(s.arrays(actor_fields=s.proposal[-1] is not None)),p['full_bytes'])
            except BaseException:save(OUT/'partial_error.json',dict(error=traceback.format_exc()))
    finally:
        if s:s.close()
        if w:w.close()
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=calls,cycles=cycles,teachers=cohorts,update_blocks=updates,result=result,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=calls));N.manifest(OUT)
    assert status=='COMPLETE'
if __name__=='__main__':main()
