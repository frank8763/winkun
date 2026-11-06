"""Actual closed-loop sampling, fixed physical-prefix curriculum and evaluation."""
from t395_learning import *
import fcntl,multiprocessing as mp

def main():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);N.configure(OUT)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=start+p['main_seconds'];learnend=start+p['learning_seconds'];phase='START';last=0.
    counts=mp.get_context('spawn').Array('q',1);commands=0;rows=[];replays=[];rounds=[];calls=dict(live=0,shadow=0)
    q,opt=learner(base,p);refs=datasets(q);w=s=None;result={};status='FAILED';active_batch=[]
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    outside=N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)
    assert outside+p['final_bytes']<=1024**3
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,learning_deadline=learnend,native_deadline=deadline))
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_main=True))
    def guard(physical=False):
        nonlocal last
        check_stop()
        assert time.time()<deadline-5 and commands<=p['max_commands'] and counts[0]<=p['max_physics']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes'])
            assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)==outside
            save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=counts[0],commands=commands,updates=calls,rounds=len(rounds),native_deadline=deadline))
    def learn_guard():assert time.time()<learnend;guard()
    def charge():counts[0]+=1;assert counts[0]<=p['max_physics']
    def record(a,r,role,path=None,reference=None):
        assert r['complete'] and not r['administrative_censor']
        rec=dict(role=role,job_id=r.get('job_id',role),physics_steps=r['physics_steps'],success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],
                 array_digest=digest(a),integration_sha256=digest(a['final_native_integration']),policy_hash=q.phash(),complete=True)
        if reference is not None:
            assert rec['array_digest']==reference['array_digest'] and rec['integration_sha256']==reference['integration_sha256']
            rec.update(exact=True,independent_sample=False);replays.append(rec);file='replays.jsonl'
        else:
            path=path or OUT/'episodes'/f'{len(rows):04}.npz'
            compact={k:a[k] for k in ['actions','native_counts','end_flags']}
            N.write_npz(path,N.S.packed(a) if role.startswith('train') else compact,p['full_bytes'] if role.startswith('train') else 5*1024)
            rec.update(path=str(path),sha256=sha(path));rows.append(rec);file='cases.jsonl'
        with (OUT/file).open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
        guard();return rec
    def session(first,seed,role,reference=None):
        nonlocal commands,s
        guard();assert commands+1<=p['max_commands'] and counts[0]+40000<=p['max_physics'];commands+=1
        m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=p['shift_x_m']
        if reference is not None and role.startswith(('initial_mode','final_mode')):
            q.load_learned(N.S.read_model(PARENT if role.startswith('initial_mode') else OUT/'candidate.pt.gz'))
        q.sample=False;q.rng=np.random.default_rng(np.random.SeedSequence(seed));q.reset_noise()
        s=Session(p['centre_left'],native,create,w,guard,charge,q);install_stop_handlers()
        for k in range(first):
            s.step(forced=prior['actions'][k],owner=1)
            assert not s.done(),'Successful fixed prefix terminated unexpectedly'
            for j,name in enumerate(N.R.NAMES):assert equal(np.asarray(s.rows[-1][j]),prior[name][k]),(k,name)
        q.sample=role.startswith('train')
        while not s.done():s.step(owner=0)
        a=s.arrays();a['hold_sha256']=np.asarray(s.digests,np.uint8);a['ownership']=np.asarray(s.owner,np.uint8)
        a['task_goal_xy']=s.goal.copy();a['scene_sha256']=np.frombuffer(bytes.fromhex(s.scene),np.uint8).copy()
        r=s.result();rec=record(a,r,role,reference=reference);s.close();s=None;q.sample=False
        return a,rec
    def evaluate(jobs,role,reference=None,readback=False):
        nonlocal commands,phase
        phase=role;items=[]
        if readback:q.load_learned(N.S.read_model(OUT/'candidate.pt.gz'))
        for j in jobs:
            guard();cap=j['case']['step_cap'];assert commands+1<=p['max_commands'] and counts[0]+cap<=p['max_physics'];commands+=1
            m.ACTIVE=copy.deepcopy(original);m.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',original['circle_shift_x_m'])
            q.sample=False;q.reset_noise();job=dict(j,source_reset_id=j['id'],role=role,arm='single_actor')
            # Use the same per-physical-step cancellation/budget guard as new
            # training, rather than the older unguarded whole-case evaluator.
            s=Session(job,native,create,w,guard,charge,q);install_stop_handlers()
            try:
                while not s.done():s.step()
                r,a=s.result(),s.arrays();r['job_id']=j['id']
            except BaseException:
                if s.rows:N.write_npz(OUT/'partial.npz',N.S.packed(s.arrays()),p['full_bytes'])
                save(OUT/'partial_case.json',dict(role=role,job_id=j['id'],actual_physics=s.steps,recorded_physics=sum(v[2] for v in s.rows),administrative_censor=True))
                raise
            finally:s.close()
            if not r['complete']:
                N.write_npz(OUT/'partial.npz',N.S.packed(a),p['full_bytes']);raise TimeoutError('Censored native validation')
            items.append(record(a,r,role,reference=None if reference is None else reference[j['id']]))
        return items
    def count(items):return dict(valid=len(items),success=sum(v['success'] for v in items),contact=sum(v['collision'] for v in items),timeout=sum(v['timeout'] for v in items),invalid=0,censored=0)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T395 external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T395 native wall deadline')));signal.alarm(p['main_seconds']-2)
    try:
        m,native,create,w,_=N.S.P.C.B.setup(OUT/'native',base);install_stop_handlers();original=copy.deepcopy(m.ACTIVE);commands+=1
        prior=N.S.arrays(REFERENCE)
        old_centres=[j for j in base['final_old_jobs'] if j['id'].startswith('dev_task') and j['id'].endswith('_13')]
        assert len(old_centres)==8
        result['initial_old8']=count(evaluate(old_centres,'initial_old8'))
        result['initial_old275']=None
        # Parent original-start main centres are already preserved T393 data.
        # This round directly tests the approved 52--56 second recovery segment.
        _,initial_mode=session(280,[p['seed'],999,0],'initial_mode56')
        session(280,[p['seed'],999,0],'initial_mode56_readback',reference=initial_mode)
        result['initial_mode56']=initial_mode
        level=0;rejected=0
        for rnd in range(p['rounds']):
            assert time.time()+120<learnend
            first=p['prefix_holds'][level];phase=f'ROUND_{rnd}_PREFIX_{first}';active_batch=[];before=state(q,opt)
            for k in range(4):
                seed=[p['seed'],rnd,k];a,r=session(first,seed,f'train_r{rnd}_{k}');active_batch.append(a)
                again,rr=session(first,seed,f'train_r{rnd}_{k}_exact',reference=r)
                assert digest(a)==digest(again)
            learn_guard();shadow,shadowopt=learner(base,p);load(shadow,shadowopt,before)
            def event(which):
                def emit(where,k):
                    if where=='AFTER':calls[which]+=1
                    with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([rnd,which,where,k])+'\n')
                return emit
            block=OUT/'blocks'/str(rnd);block.mkdir()
            # Exact before Actor+Adam is reconstructible from parent and all
            # previous raw blocks; avoid a second binary copy per round.
            save(block/'before_hashes.json',statehash(q,opt))
            try:
                live=fit(q,opt,active_batch,first,p,refs,learn_guard,event('live'))
                other=fit(shadow,shadowopt,active_batch,first,p,refs,learn_guard,event('shadow'))
                assert live==other and statehash(q,opt)==statehash(shadow,shadowopt)
                accepted=live['numeric_accept'];proposed=export(q,opt,block/'proposed.pt.gz')
                if not accepted:load(q,opt,before)
                assert accepted or digest(state(q,opt))==digest(before)
            except BaseException:
                load(q,opt,before);save(block/'interrupted_rollback.json',dict(exact=digest(state(q,opt))==digest(before)));raise
            suffix,centre=session(first,[p['seed'],rnd,100],'course_centre')
            success=sum(bool(a['end_flags'][-1,1]) and not bool(a['end_flags'][:,0].any()) for a in active_batch)
            promote=accepted and success>=2 and centre['success'];rejected=0 if accepted else rejected+1
            row=dict(round=rnd,prefix_holds=first,policy_before=digest(before['learned']),sampling_safe=success,sampling_count=4,proposal=live,
                     proposed_model=proposed,accepted=accepted,course_centre=centre,promote=promote,level_before=level)
            if promote and level<len(p['prefix_holds'])-1:level+=1
            row['level_after']=level;rounds.append(row);save(block/'result.json',row);save(OUT/'rounds.json',rounds)
            if rejected>=2:break
        result['model']=export(q,opt,OUT/'candidate.pt.gz')
        result.update(old275=None,main54=None,fresh16=None,reload78=None,stable_new070_80=False,original_control=None)
        final_prefix=[]
        for first in p['prefix_holds']:
            _,row=session(first,[p['seed'],999,first],'final_mode_prefix'+str(first));final_prefix.append(row)
            session(first,[p['seed'],999,first],'final_mode_prefix'+str(first)+'_readback',reference=row)
        result['final_prefix_modes']=final_prefix
        result['final_old8']=count(evaluate(old_centres,'final_old8'))
        centres=evaluate(p['centres'],'final_main_centres');result['main_centres']=count(centres)
        evaluate(p['centres'],'final_main_centres_readback',{v['job_id']:v for v in centres},True)
        result['original_control']=count(evaluate(base['original_jobs'],'original057_56_control'))
        # Independent approval is needed after this finite recovery-segment
        # pilot; full54/275/fresh16/reload78 are explicitly not scheduled here.
        status='COMPLETE'
    except BaseException:
        if s is not None and s.rows:
            try:N.write_npz(OUT/'partial.npz',N.S.packed(s.arrays()),p['full_bytes'])
            except BaseException:save(OUT/'partial_save_error.json',dict(error=traceback.format_exc()))
        save(OUT/'failure.json',dict(phase=phase,error=traceback.format_exc(),physics=counts[0],commands=commands,updates=calls))
    finally:
        if s is not None:s.close()
        if w is not None:w.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        summary=dict(state=status,pid=os.getpid(),elapsed_s=time.time()-start,physics=counts[0],commands=commands,rounds=rounds,updates=calls,result=result,
                     native_setup_commands=1,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=status,updated_epoch=time.time(),physics=counts[0],commands=commands,updates=calls))
        N.manifest(OUT)
    assert status=='COMPLETE'

def audit():
    N.offline();p=read(OUT/'protocol.json');base=verify(p);summary=read(OUT/'summary.json')
    assert summary['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    assert not Path('/proc',str(summary['pid'])).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);refs=datasets(q);calls=0
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    for row in cases:
        assert sha(row['path'])==row['sha256'];a=N.load_npz(row['path'])
        assert int(a['native_counts'].sum())==row['physics_steps'] and bool(a['end_flags'][-1,1])==row['success']
    for r in summary['rounds']:
        batch=[N.S.arrays(next(v['path'] for v in cases if v['role']==f'train_r{r["round"]}_{k}')) for k in range(4)]
        before=state(q,opt);assert digest(before['learned'])==r['policy_before']
        for k,a in enumerate(batch):audit_training(q,a,r['prefix_holds'],[p['seed'],r['round'],k],p)
        actual=fit(q,opt,batch,r['prefix_holds'],p,refs,lambda:None,lambda *_:None);calls+=p['epochs']
        assert actual==r['proposal'] and statehash(q,opt)==r['proposed_model']['hashes']
        if not r['accepted']:load(q,opt,before)
    assert statehash(q,opt)==summary['result']['model']['hashes']
    assert summary['updates']==dict(live=calls,shadow=calls)
    assert summary['physics']==sum(v['physics_steps'] for v in [*cases,*reps])
    assert summary['commands']==len(cases)+len(reps)+summary['native_setup_commands']
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    save(OUT/'audit.json',dict(state='PASS',native=0,new_training=0,discarded_historical_Adam=calls,Actor_and_Adam_exact=True,raw_ledger_exact=True,
        physics=summary['physics'],commands=summary['commands'],stable_new070_80=summary['result']['stable_new070_80']))
    N.manifest(OUT);print('T395_AUDIT_PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
