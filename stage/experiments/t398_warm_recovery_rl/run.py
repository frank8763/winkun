"""Finite teacher queries, separate student PPO, old-capability acceptance."""
from t398_learning import *
import fcntl

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();learnend=start+p['learning_seconds'];deadline=start+p['main_seconds']
    q,opt=learner(base,p);oldrefs=old_references(q,base,lambda:None);prior=N.S.arrays(REFERENCE);s=w=None;phase='START';last=0.;physics=commands=0
    cases=[];replays=[];rounds=[];cohorts=[];calls=dict(live=0,shadow=0);state_final='FAILED';result={};corrections=[KNOWN_BRIDGE]
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,learning_deadline=learnend,native_deadline=deadline))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_main=True))
    def guard(physical=False):
        nonlocal last
        check_stop();assert time.time()<deadline-5 and commands<=p['max_commands'] and physics<=p['max_physics']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,commands=commands,updates=calls,rounds=len(rounds),deadline=deadline))
    def lg():assert time.time()<learnend-5;guard()
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    def command(label,fn):
        nonlocal commands
        guard();commands+=1;before=physics;t=time.time();error=None
        try:return fn()
        except BaseException:
            error=traceback.format_exc();raise
        finally:
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,label=label,physics=physics-before,elapsed_s=time.time()-t,error=error),separators=(',',':'))+'\n')
    def arrays():
        a=s.arrays();a.update(hold_sha256=np.asarray(s.digests,np.uint8),ownership=np.asarray(s.owner,np.uint8),task_goal_xy=s.goal.copy(),scene_sha256=np.frombuffer(bytes.fromhex(s.scene),np.uint8).copy());return a
    def construct(job):
        nonlocal s
        guard();m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=job.get('circle_shift_x_m',original['circle_shift_x_m'])
        q.sample=False;q.reset_noise();s=Session(job,native,create,w,guard,charge,q);install_stop_handlers()
    def prefix(first):
        for i in range(first):
            s.step(forced=prior['actions'][i],owner=1)
            assert not s.done()
            for j,name in enumerate(N.R.NAMES):assert equal(np.asarray(s.rows[-1][j]),prior[name][i]),(i,name)
    def record(a,role,first=0,full=False,reference=None):
        rec=dict(role=role,policy_hash=q.phash(),physics_steps=s.steps,success=bool(a['end_flags'][-1,1]) and not bool(a['end_flags'][:,0].any()),collision=bool(a['end_flags'][:,0].any()),timeout=bool(a['end_flags'][-1,2]),remaining_m=float(np.linalg.norm(a['next_poses'][-1,:2]-s.goal)),array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),prefix_holds=first)
        if reference is None:
            path=OUT/'episodes'/f'{len(cases):04}.npz';data=N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']}
            N.write_npz(path,data,p['full_bytes'] if full else 5*1024);rec.update(path=str(path),sha256=sha(path));cases.append(rec);file='cases.jsonl'
        else:
            assert rec['array_digest']==reference['array_digest'] and rec['integration_sha256']==reference['integration_sha256']
            rec.update(exact=True,independent_sample=False);replays.append(rec);file='replays.jsonl'
        with (OUT/file).open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
        return rec
    def episode(job,role,first=0,seed=None,reference=None,readback=False):
        nonlocal s,phase
        phase=role
        def body():
            nonlocal s
            if readback:q.load_learned(N.S.read_model(OUT/'candidate.pt.gz'))
            construct(job);prefix(first)
            q.sample=seed is not None;q.rng=None if seed is None else np.random.default_rng(np.random.SeedSequence(seed))
            while not s.done():s.step(owner=0)
            a=arrays();r=record(a,role,first,seed is not None,reference);s.close();s=None;q.sample=False;return a,r
        return command(role,body)
    def counts(rs):return dict(valid=len(rs),success=sum(r['success'] for r in rs),contact=sum(r['collision'] for r in rs),timeout=sum(r['timeout'] for r in rs),invalid=0,censored=0)
    def old8(role):return [episode(j,role+'_'+j['id'])[1] for j in old]
    def teacher(cycle):
        nonlocal s,phase
        folder=OUT/'teachers'/str(cycle);folder.mkdir(parents=True);data=teacher_buffer(p);begin=physics;cbegin=commands
        phase=f'TEACHER_{cycle}_CURRENT_STUDENT'
        def initial():
            construct(p['centre_left']);prefix(p['prefix_hold'])
            for _ in range(p['learner_holds']):s.step(owner=0);assert not s.done()
            N.write_npz(folder/'teacher_full.npz',N.S.packed(arrays()),p['full_bytes'])
        command(phase,initial);snapshot=s.capture();snapshot_hash=N.C.snapshot_hash(s)
        mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['teacher_sd']),(16,2)).copy();best=None;bestid=None;bestcost=float('inf');found=False;traces=[];query_count=0
        for gen in range(2):
            ids=[]
            for j,theta in enumerate(teacher_candidates(mean,sd,best,cycle,gen,p)):
                lg();idx=12*gen+j;u=teacher_controls(theta,prior['actions'],p);data['coefficients'][idx]=theta;data['controls'][idx]=u
                for rep in range(2):
                    phase=f'TEACHER_{cycle}_QUERY_{idx}_REP_{rep}'
                    def query():
                        s.restore(snapshot);q.sample=False
                        for t,action in enumerate(u):
                            row,hd=s.step(action,owner=1,actor_active=False)
                            if rep==0:
                                data['next_poses'][idx,t]=row[5];data['counts'][idx,t]=row[2];data['flags'][idx,t]=row[6];data['hold_sha256'][idx,t]=hd
                            else:
                                assert equal(data['next_poses'][idx,t],row[5]) and data['counts'][idx,t]==row[2] and equal(data['flags'][idx,t],np.asarray(row[6],np.uint8)) and equal(data['hold_sha256'][idx,t],hd)
                            if s.done():break
                        assert s.done();n=t+1;rem=float(s.env._goal_state()[1]);cost=objective(u,n,s.rows[-1][6],rem);integ=N.K.R.integration(s.env)
                        if rep==0:data['lengths'][idx]=n;data['costs'][idx]=cost;data['remaining_m'][idx]=rem;data['final_integration'][idx]=integ
                        else:assert data['lengths'][idx]==n and data['costs'][idx]==cost and equal(data['final_integration'][idx],integ)
                    command(phase,query)
                N.write_npz(folder/'search.npz',data,p['search_bytes']);ids.append(idx);query_count+=1
                score=float(data['costs'][idx])
                if bestid is None or (score,idx)<(bestcost,bestid):best=theta.copy();bestid=idx;bestcost=score
                found=bool(data['flags'][idx,int(data['lengths'][idx])-1,1]) and not bool(data['flags'][idx,:,0].any())
                if found:break
            elite=sorted(ids,key=lambda i:(float(data['costs'][i]),i))[:3]
            if not found:
                assert len(elite)==3;mean=.5*mean+.5*data['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['teacher_min_sd']),.5*sd+.5*data['coefficients'][elite].std(0))
            traces.append(dict(generation=gen,ids=ids,elite=elite,mean_after=mean.tolist(),sd_after=sd.tolist(),best_id=int(bestid),goal_found=found))
            if found:break
        cohort=dict(cycle=cycle,policy_hash=q.phash(),snapshot_sha256=snapshot_hash,teacher_start_hold=p['teacher_start_hold'],queries=query_count,max_queries=24,query_exact_repeats=query_count,full_teacher_safe=found,short_bridge_safe=False,traces=traces,best_id=int(bestid),duplicate_first_two_controls=bool(equal(data['controls'][0],data['controls'][1])) if query_count>1 else False)
        if found:
            u=data['controls'][bestid];saved={}
            for kind in ['teacher_full','bridge']:
                for rep in range(2):
                    phase=f'TEACHER_{cycle}_{kind}_REP_{rep}'
                    def verified():
                        s.restore(snapshot);q.sample=False
                        n=int(data['lengths'][bestid]) if kind=='teacher_full' else p['bridge_holds']
                        for t in range(n):s.step(u[t],owner=1)
                        if kind=='bridge':
                            while not s.done():s.step(owner=0)
                        assert s.done();a=arrays()
                        if rep==0:saved[kind]=a;N.write_npz(folder/(kind+'.npz'),N.S.packed(a),p['full_bytes'])
                        else:assert digest(a)==digest(saved[kind])
                    command(phase,verified)
            assert saved['teacher_full']['end_flags'][-1,1] and not saved['teacher_full']['end_flags'][:,0].any()
            cohort['short_bridge_safe']=bool(saved['bridge']['end_flags'][-1,1]) and not bool(saved['bridge']['end_flags'][:,0].any())
            cohort['full_hash']=digest(saved['teacher_full']);cohort['bridge_hash']=digest(saved['bridge'])
        cohort.update(physics=physics-begin,commands=commands-cbegin,search_hash=digest(data),training_labels_allowed=bool(found and cohort['short_bridge_safe']))
        s.close();s=None;save(folder/'result.json',cohort);cohorts.append(cohort);save(OUT/'teachers.json',cohorts)
        return folder/'bridge.npz' if cohort['training_labels_allowed'] else None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T398 native wall deadline')));signal.alarm(p['main_seconds']-2)
    try:
        def setup():return N.S.P.C.B.setup(OUT/'native',base)
        m,native,create,w,_=command('native_setup',setup);install_stop_handlers();original=copy.deepcopy(m.ACTIVE)
        old=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')];assert len(old)==8
        baseline=read(ROOT/'stage/runs/t396_iterative_recovery/summary.json')
        assert q.phash()==baseline['result']['model']['policy_hash']
        result['initial_old8']=baseline['result']['initial_old8'];assert result['initial_old8']['success']==8
        result['initial_mode']=baseline['result']['initial_mode']
        result['initial_baseline_reused']=dict(policy_hash=q.phash(),source='T396 frozen same T393',new_native=0)
        save(OUT/'old_reference_contract.json',dict(episodes=len(oldrefs),rows=sum(len(g['x']) for g in oldrefs),targets='T393 raw location on historical T255 causal episodes, not recovery expert',sha256=digest([(g['x'],g['target']) for g in oldrefs])))
        before=state(q,opt);shadow,so=learner(base,p);load(shadow,so,before)
        warmfolder=OUT/'warm';warmfolder.mkdir()
        def warm_event(which):
            def emit(where,k):
                if where=='AFTER':calls[which]+=1
                with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(['warm',which,where,k])+'\n')
            return emit
        phase='SUPERVISED_WARM_200'
        try:
            live=warm_fit(q,opt,groups(q,p,corrections),oldrefs,p,lg,warm_event('live'))
            other=warm_fit(shadow,so,groups(shadow,p,corrections),oldrefs,p,lg,warm_event('shadow'))
            assert live==other and statehash(q,opt)==statehash(shadow,so)
            proposed=export(q,opt,warmfolder/'proposed.pt.gz');tested=old8('warm_proposal_old8')
            warm_accepted=all(r['success'] for r in tested)
            if not warm_accepted:load(q,opt,before);assert digest(state(q,opt))==digest(before)
        except BaseException:
            load(q,opt,before);save(warmfolder/'interrupted_rollback.json',dict(exact=digest(state(q,opt))==digest(before)));raise
        warm=dict(proposal=live,proposed_model=proposed,old8=counts(tested),accepted=warm_accepted,retained_hashes=statehash(q,opt))
        # The supervised optimizer history is never silently carried into PPO.
        opt=torch.optim.Adam(N.trainable(q),lr=p['lr'])
        warm['post_stage_fresh_Adam_hashes']=statehash(q,opt);save(warmfolder/'result.json',warm);result['warm']=warm
        result['post_warm_mode']=[episode(p['centre_left'],'post_warm_mode_'+str(first),first)[1] for first in [265,270]]
        reason='TWO_CYCLES_COMPLETE' if warm_accepted else 'WARM_REJECTED_EXACT_ACTOR_ADAM_ROLLBACK_NO_RL'
        for cycle in range(2 if warm_accepted else 0):
            assert time.time()+900<learnend
            before=state(q,opt);file=teacher(cycle)
            if file is None:reason='NO_VERIFIED_SHORT_BRIDGE';break
            corrections.append(file);batch=[]
            for k in range(4):
                seed=[p['seed'],cycle,k];a,r=episode(p['centre_left'],f'train_r{cycle}_{k}',265,seed);batch.append(a)
                episode(p['centre_left'],f'train_r{cycle}_{k}_exact',265,seed,r)
            lg();refs=groups(q,p,corrections);shadow,so=learner(base,p);load(shadow,so,before)
            folder=OUT/'blocks'/str(cycle);folder.mkdir();save(folder/'before_hashes.json',statehash(q,opt))
            def event(which):
                def emit(where,k):
                    if where=='AFTER':calls[which]+=1
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([cycle,which,where,k])+'\n')
                return emit
            try:
                live=fit(q,opt,batch,265,p,refs,oldrefs,lg,event('live'));other=fit(shadow,so,batch,265,p,groups(shadow,p,corrections),oldrefs,lg,event('shadow'))
                assert live==other and statehash(q,opt)==statehash(shadow,so);proposed=export(q,opt,folder/'proposed.pt.gz')
                tested=old8(f'proposal_{cycle}_old8') if live['numeric_accept'] else []
                accepted=live['numeric_accept'] and len(tested)==8 and all(r['success'] for r in tested)
                if not accepted:load(q,opt,before);assert digest(state(q,opt))==digest(before)
            except BaseException:
                load(q,opt,before);save(folder/'interrupted_rollback.json',dict(exact=digest(state(q,opt))==digest(before)));raise
            modes=[episode(p['centre_left'],f'round_{cycle}_mode_{first}',first)[1] for first in [265,270]]
            row=dict(cycle=cycle,corrections=[str(v) for v in corrections],proposal=live,proposed_model=proposed,old8=counts(tested) if tested else None,accepted=accepted,modes=modes,retained_hashes=statehash(q,opt))
            rounds.append(row);save(folder/'result.json',row);save(OUT/'rounds.json',rounds)
            if not accepted:reason='CANDIDATE_REJECTED_EXACT_ACTOR_ADAM_ROLLBACK';break
        result['reason']=reason;result['model']=export(q,opt,OUT/'candidate.pt.gz')
        result['final_mode']=[]
        for first in [265,270]:
            _,r=episode(p['centre_left'],'final_mode_'+str(first),first);result['final_mode'].append(r)
            episode(p['centre_left'],'final_mode_'+str(first)+'_readback',first,reference=r,readback=True)
        result['final_old8']=counts(old8('final_old8'))
        centres=[]
        for j in p['centres']:
            _,r=episode(j,'final_main_'+j['id']);centres.append(r);episode(j,'final_main_'+j['id']+'_readback',reference=r,readback=True)
        result.update(main_centres=counts(centres),old275=None,main54=None,fresh16=None,reload78=None,stable_new070_80=False)
        state_final='COMPLETE'
    except BaseException:
        save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=physics,commands=commands,updates=calls))
        if s is not None and s.rows:
            try:N.write_npz(OUT/'partial.npz',N.S.packed(s.arrays(actor_fields=s.proposal[-1] is not None)),p['full_bytes'])
            except BaseException:save(OUT/'partial_save_error.json',dict(error=traceback.format_exc()))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        events=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()] if (OUT/'optimizer_calls.jsonl').exists() else []
        split={stage:{which:sum(v[0]==stage and v[1]==which and v[2]=='AFTER' for v in events) for which in ['live','shadow']} for stage in ['warm',0,1]}
        summary=dict(state=state_final,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,updates=calls,rounds=rounds,teachers=cohorts,result=result,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False,matched_fixed_demo_control_not_run=True,actual_update_split=split)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state_final,updated_epoch=time.time(),physics=physics,commands=commands,updates=calls));N.manifest(OUT)
    assert state_final=='COMPLETE'

if __name__=='__main__':globals()[sys.argv[1]]()
