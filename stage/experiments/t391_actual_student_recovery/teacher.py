"""One finite training-only oracle query at the actual new learner state."""
from recovery_common import *
T=OUT/'teacher'


def controls(theta,reference,p):
    h=p['remaining_holds'];axis=np.arange(h);knots=np.linspace(0,h-1,16)
    v=np.stack([np.interp(axis,knots,theta[:,j]) for j in range(2)],1)
    clock=p['prefix_holds']+axis
    phase=np.maximum.accumulate(clock+5*np.clip(v[:,1],-4.,4.))
    phase=np.clip(phase,0,len(reference)-1)
    u=np.stack([np.interp(phase,np.arange(len(reference)),reference[:,j]) for j in range(2)],1)
    u[:,0]+=v[:,0]
    return np.clip(u,-1,1).astype(np.float32)


def candidates(mean,sd,best,it,p):
    z=np.random.default_rng(np.random.SeedSequence([p['seed'],it])).normal(size=(5,16,2))
    result=[mean.copy(),np.zeros((16,2)) if best is None else best.copy()]
    for row in z:result.extend([mean+row*sd,mean-row*sd])
    return np.asarray(result)


def objective(u,n,flags,remaining):
    contact,goal,timeout=map(bool,flags)
    return 1000.*contact+500.*timeout-1000.*(goal and not contact)+10.*remaining+.01*float(np.square(np.diff(u[:n].astype(float),axis=0)).sum())


def compact(p):
    n,h=36,p['remaining_holds']
    return dict(coefficients=np.zeros((n,16,2)),controls=np.zeros((n,h,2),np.float32),
                next_poses=np.full((n,h,3),np.nan),counts=np.zeros((n,h),np.uint8),
                flags=np.zeros((n,h,3),np.uint8),hold_sha256=np.zeros((n,h,32),np.uint8),
                lengths=np.zeros(n,np.uint16),costs=np.full(n,np.nan),remaining_m=np.full(n,np.nan),final_integration=np.full((n,203),np.nan))


def main():
    N.offline();p=read(T/'protocol.json');base=verify(p);N.configure(T)
    assert not (T/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=min(start+p['main_seconds'],p['absolute_deadline'])
    physics=commands=0;last=0.;state='FAILED';phase='SETUP';s=w=None;active=False;passes=[]
    data=compact(p);source=N.S.arrays(SOURCE);reference=N.S.arrays(REFERENCE)['actions'];actual=None
    first=p['prefix_holds']
    def guard(physical=False):
        nonlocal last
        assert time.time()<deadline-10 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(T,p['final_bytes']);verify_root_capacity()
            save(T/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physics,commands=commands,deadline=deadline))
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    def command():
        nonlocal commands
        commands+=1;guard()
    def persist():N.write_npz(T/'mixed.npz',N.S.packed(s.mixed()),640*1024)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery teacher external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Recovery teacher wall deadline')))
    signal.alarm(max(1,int(deadline-start)-5))
    save(T/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        command();native,create,w,job=N.R.setup(base)
        for rep in range(2):
            phase=f'PASS{rep}_ACTUAL_STUDENT_PREFIX';command()
            s=N.R.Session(job,native,create,w,guard,charge,actor(base))
            assert equal(s.initial,source['initial_native_integration'])
            for i in range(first):
                s.step()
                for j,key in enumerate(N.NAMES):assert equal(np.asarray(s.rows[-1][j]),source[key][i]),key
                assert all(equal(a,b) for a,b in zip(s.hist[-1],[source[k][i] for k in ['history_poses','history_goals','history_times']]))
                assert equal(s.hb[-1],source['hidden_before'][i])
                for key,v in s.proposal[-1].items():assert equal(v,source[key][i]),key
            assert s.steps==first*100 and not s.done()
            snapshot=s.capture();snapshot_hash=N.C.snapshot_hash(s)
            if rep==0:persist()
            else:assert snapshot_hash==passes[0]['actual_student_snapshot_hash']
            mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['initial_sd']),(16,2)).copy()
            best=None;best_id=None;best_cost=float('inf');traces=[];queries=0;goal=False
            for it in range(3):
                ids=[]
                for j,theta in enumerate(candidates(mean,sd,best,it,p)):
                    idx=it*12+j;phase=f'PASS{rep}_ITER{it}_QUERY{j}';command();s.restore(snapshot)
                    u=controls(theta,reference,p);active=True
                    if rep==0:data['coefficients'][idx]=theta;data['controls'][idx]=u
                    for t in range(p['remaining_holds']):
                        row,hd=s.step(u[t],owner=1,actor_active=False)
                        if rep==0:
                            data['next_poses'][idx,t]=row[5];data['counts'][idx,t]=row[2]
                            data['flags'][idx,t]=row[6];data['hold_sha256'][idx,t]=hd
                        else:
                            assert equal(data['next_poses'][idx,t],row[5]) and data['counts'][idx,t]==row[2]
                            assert equal(data['flags'][idx,t],np.asarray(row[6],np.uint8)) and equal(data['hold_sha256'][idx,t],hd)
                        if s.done():break
                    assert s.done();n=t+1;remaining=float(s.env._goal_state()[1]);score=objective(u,n,s.rows[-1][6],remaining)
                    integ=N.K.R.integration(s.env)
                    if rep==0:
                        data['lengths'][idx]=n;data['costs'][idx]=score;data['remaining_m'][idx]=remaining;data['final_integration'][idx]=integ
                        N.write_npz(T/'search.npz',data,1152*1024)
                    else:
                        assert equal(theta,data['coefficients'][idx]) and equal(u,data['controls'][idx])
                        assert data['lengths'][idx]==n and score==data['costs'][idx] and equal(integ,data['final_integration'][idx])
                    ids.append(idx);queries+=1;active=False
                    if best_id is None or (score,idx)<(best_cost,best_id):best=theta.copy();best_id=idx;best_cost=score
                    goal=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision'))
                    if goal:break
                elite=sorted(ids,key=lambda k:(float(data['costs'][k]),k))[:3]
                before_mean=mean.copy();before_sd=sd.copy()
                if not goal:
                    assert len(elite)==3
                    mean=.5*mean+.5*data['coefficients'][elite].mean(0)
                    sd=np.maximum(np.asarray(p['minimum_sd']),.5*sd+.5*data['coefficients'][elite].std(0))
                traces.append(dict(iteration=it,queries=ids,elite=elite,mean_before=before_mean.tolist(),sd_before=before_sd.tolist(),
                                   mean_after=mean.tolist(),sd_after=sd.tolist(),best_id=int(best_id),best_cost=float(best_cost),safe_GOAL=goal))
                if goal:break
            phase=f'PASS{rep}_ACTUAL_VERIFIED_RECOVERY';command();s.restore(snapshot)
            for t in range(int(data['lengths'][best_id])):
                row,hd=s.step(data['controls'][best_id,t],owner=1)
                assert equal(row[5],data['next_poses'][best_id,t]) and row[2]==data['counts'][best_id,t]
                assert equal(np.asarray(row[6],np.uint8),data['flags'][best_id,t])
                if rep==0:
                    if t%10==0 or s.done():persist()
                else:assert equal(hd,actual['hold_sha256'][len(s.rows)-1])
                if s.done():break
            assert s.done() and equal(N.K.R.integration(s.env),data['final_integration'][best_id])
            a=s.mixed()
            if rep==0:persist();actual=copy.deepcopy(a)
            else:assert hashes(a)==hashes(actual)
            passes.append(dict(pass_index=rep,actual_student_snapshot_hash=snapshot_hash,prefix_holds=first,queries=queries,best_id=int(best_id),
                               goal_found=goal,mixed=s.result(),iterations=traces,mixed_hashes=hashes(a),search_hashes=hashes(data),independent_sample=rep==0))
            save(T/'passes.json',passes);s.close();s=None
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if s:
            try:
                if active:N.write_npz(T/'partial_candidate.npz',N.S.packed(s.arrays(actor_fields=False)),640*1024)
                N.write_npz(T/'interruption.npz',dict(packet=s.packet(),integration=N.K.R.integration(s.env),steps=np.asarray(s.steps)),32*1024)
            except BaseException:save(T/'partial_error.json',dict(error=traceback.format_exc()))
        save(T/'failure.json',dict(error=error,phase=phase,physics=physics,commands=commands))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0)
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,Actor=0,optimizer=0,
                     passes=passes,all_workers_stopped=True,teacher_safe_GOAL=bool(state=='COMPLETE' and passes[0]['mixed']['success']),
                     autonomous_six_stable=False,student_parent_sha256=sha(PARENT),
                     files={str(f.relative_to(T)):sha(f) for f in T.rglob('*') if f.is_file() and f.name!='status.json'})
        save(T/'summary.json',summary);save(T/'status.json',dict(state=state,updated_epoch=time.time(),physics=physics,commands=commands))
        N.capacity(T,p['final_bytes']);verify_root_capacity()
    assert state=='COMPLETE','Shared engineering failure stops this experiment'


def verify_root_capacity():
    assert N.K.S.used(OUT)<=read(OUT/'protocol.json')['final_bytes']


def audit():
    N.offline();p=read(T/'protocol.json');base=verify(p);s=read(T/'summary.json');N.stopped(T)
    assert s['state']=='COMPLETE' and not (T/'audit.json').exists()
    f,g=s['passes'];a=N.S.arrays(T/'mixed.npz');d=N.load_npz(T/'search.npz');source=N.S.arrays(SOURCE);ref=N.S.arrays(REFERENCE)['actions'];first=p['prefix_holds']
    assert f['iterations']==g['iterations'] and f['best_id']==g['best_id'] and f['queries']==g['queries']
    for key in N.NAMES:assert equal(a[key][:first],source[key][:first]),key
    mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['initial_sd']),(16,2)).copy();best=None;best_id=None;best_cost=float('inf')
    for it,rec in enumerate(f['iterations']):
        proposal=candidates(mean,sd,best,it,p)
        for idx in rec['queries']:
            theta=proposal[idx%12];u=controls(theta,ref,p);n=int(d['lengths'][idx])
            assert n and equal(theta,d['coefficients'][idx]) and equal(u,d['controls'][idx])
            assert not d['flags'][idx,:n-1].any() and d['flags'][idx,n-1].any()
            score=objective(u,n,d['flags'][idx,n-1],d['remaining_m'][idx]);assert score==d['costs'][idx]
            if best_id is None or (score,idx)<(best_cost,best_id):best=theta.copy();best_id=idx;best_cost=score
        elite=sorted(rec['queries'],key=lambda k:(float(d['costs'][k]),k))[:3];assert elite==rec['elite'] and best_id==rec['best_id']
        if not rec['safe_GOAL']:
            mean=.5*mean+.5*d['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['minimum_sd']),.5*sd+.5*d['coefficients'][elite].std(0))
        assert equal(mean,np.asarray(rec['mean_after'])) and equal(sd,np.asarray(rec['sd_after']))
    assert best_id==f['best_id'];n=int(d['lengths'][best_id]);assert len(a['states'])==first+n
    assert equal(a['actions'][first:],d['controls'][best_id,:n]) and np.all(a['ownership'][:first]==0) and np.all(a['ownership'][first:]==1)
    assert hashes(a)==f['mixed_hashes']==g['mixed_hashes'] and hashes(d)==f['search_hashes']==g['search_hashes']
    physical=2*(first*100+int(d['counts'].sum())+int(a['native_counts'][first:].sum()))
    assert physical==s['physics']<=p['max_physics'] and s['commands']==1+2*(2+f['queries'])<=p['max_commands']
    q=actor(base);hidden=None;goal=np.asarray(read(N.R.SOURCE/'cases.json')[0]['goal_xy'])
    for i,x in enumerate(a['states']):
        assert np.abs(N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        before=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
        with torch.no_grad():mode,_,hidden,_=q.step(torch.tensor(x[None]),hidden)
        assert equal(before,a['hidden_before'][i])
        for key,v in q.tape[-1].items():assert equal(v,a['actor_proposal_'+key][i])
        if i<first:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal)
        z=N.K.R.GAMMA**int(a['native_counts'][i]);flags=a['end_flags'][i];term=bool(flags.any())
        reward=-.000002*(1-z)/(1-N.K.R.GAMMA)+.03*d0-z*.03*d1*(not term)
        if term:reward+=N.K.R.GAMMA**(int(a['native_counts'][i])-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(reward-float(a['rewards'][i]))<1e-10
    for name,h in s['files'].items():assert sha(T/name)==h,name
    save(T/'audit.json',dict(state='PASS',native=0,Actor=0,optimizer=0,all_candidates_costs_and_two_pass_arrays_exact=True,
                            actual_student_prefix_and_causal_history_exact=True,teacher_safe_GOAL=s['teacher_safe_GOAL'],
                            physics=s['physics'],commands=s['commands'],not_autonomous_Actor_success=True))
    N.manifest(T);print('T391_TEACHER_INDEPENDENT_AUDIT_PASS')


if __name__=='__main__':globals()[sys.argv[1]]()
