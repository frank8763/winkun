"""Finite full-task CEM over a demonstration action prior. Not navigation RL."""
from night_common import *

def controls(theta,human,p):
    h=380;knots=np.linspace(0,h-1,16);v=np.stack([np.interp(np.arange(h),knots,theta[:,j]) for j in range(2)],1)
    clock=20+np.arange(h)
    if p['representation']=='ACTION_RESIDUAL':
        commands=human[np.minimum(clock,len(human)-1)].astype(np.float64)+v
    else:
        phase=np.maximum.accumulate(clock+5*np.clip(v[:,1],-4.,4.));phase=np.clip(phase,0,len(human)-1)
        commands=np.stack([np.interp(phase,np.arange(len(human)),human[:,j]) for j in range(2)],1)
        commands[:,0]+=v[:,0]
    return np.clip(commands,-1,1).astype(np.float32)
def candidates(mean,sd,best,it,p):
    z=np.random.default_rng(np.random.SeedSequence([p['seed'],it])).normal(size=(5,16,2))
    result=[mean.copy(),np.zeros((16,2)) if best is None else best.copy()]
    for a in z:result.extend([mean+a*sd,mean-a*sd])
    return np.asarray(result)
def objective(commands,n,flags,remaining):
    contact,goal,timeout=map(bool,flags)
    return 1000.*contact+500.*timeout-1000.*(goal and not contact)+10.*remaining+.01*float(np.square(np.diff(commands[:n].astype(np.float64),axis=0)).sum())
def compact_search():
    return dict(coefficients=np.zeros((72,16,2)),controls=np.zeros((72,380,2),np.float32),next_poses=np.full((72,380,3),np.nan),
        counts=np.zeros((72,380),np.uint8),flags=np.zeros((72,380,3),np.uint8),hold_sha256=np.zeros((72,380,32),np.uint8),
        lengths=np.zeros(72,np.uint16),costs=np.full(72,np.nan),remaining_m=np.full(72,np.nan),final_integration=np.full((72,203),np.nan))
def main(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');configure(out);base=science(p)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not (out/'summary.json').exists()
    start=time.time();deadline=min(start+p['main_seconds'],p['native_deadline']);physics=commands=0;last=0.;phase='SETUP';state='FAILED';s=w=None;results=[];data_=compact_search();actualref=None;active=None
    def guard(physical=False):
        nonlocal last
        assert time.time()<deadline-10 and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();capacity(out,p['final_bytes']);save(out/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=physical_count(),commands=commands,deadline=deadline))
    def physical_count():return physics
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['max_physics']
    def command():
        nonlocal commands
        commands+=1;guard()
    def persist_actual():
        if s and s.rows:write_npz(out/'mixed.npz',S.packed(s.mixed()),512*1024)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full-task search external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full-task search deadline')));signal.alarm(max(1,int(deadline-start)-5))
    save(out/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        native,create,w,job=R.setup(base);source=S.arrays(R.SOURCE/'centre.npz');human=S.arrays(base['demonstration'])['actions']
        for rep in range(2):
            phase=f'PASS{rep}_ACTUAL_PREFIX4S';command();s=R.Session(job,native,create,w,guard,charge,R.actor(base))
            for i in range(20):
                s.step();assert all(equal(np.asarray(s.rows[-1][j]),source[key][i]) for j,key in enumerate(NAMES))
            assert s.steps==2000 and not s.done();snapshot=s.capture();prefix_hash=C.snapshot_hash(s)
            if rep==0:persist_actual()
            else:assert prefix_hash==results[0]['prefix_hash']
            mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['initial_sd']),(16,2)).copy();best=None;best_id=None;best_cost=float('inf');traces=[];queries=0;goal_found=False
            for it in range(6):
                proposals=candidates(mean,sd,best,it,p);iteration_ids=[]
                for j,theta in enumerate(proposals):
                    idx=it*12+j;phase=f'PASS{rep}_CEM_ITER{it}_CANDIDATE{j}';command();s.restore(snapshot);u=controls(theta,human,p);active=dict(index=idx,start_rows=20)
                    if rep==0:data_['coefficients'][idx]=theta;data_['controls'][idx]=u;write_npz(out/'search.npz',data_,2176*1024)
                    for t in range(380):
                        row,hd=s.step(u[t],owner=1,actor_active=False)
                        if rep==0:
                            data_['next_poses'][idx,t]=row[5];data_['counts'][idx,t]=row[2];data_['flags'][idx,t]=row[6];data_['hold_sha256'][idx,t]=hd
                        else:assert equal(data_['next_poses'][idx,t],row[5]) and data_['counts'][idx,t]==row[2] and equal(data_['flags'][idx,t],np.asarray(row[6],np.uint8)) and equal(data_['hold_sha256'][idx,t],hd)
                        if s.done():break
                    assert s.done(),'Full task must end in actual CONTACT/GOAL/trueTIMEOUT'
                    n=t+1;remaining=float(s.env._goal_state()[1]);score=objective(u,n,s.rows[-1][6],remaining);integration=K.R.integration(s.env)
                    if rep==0:
                        data_['lengths'][idx]=n;data_['costs'][idx]=score;data_['remaining_m'][idx]=remaining;data_['final_integration'][idx]=integration;write_npz(out/'search.npz',data_,2176*1024)
                    else:assert equal(theta,data_['coefficients'][idx]) and equal(u,data_['controls'][idx]) and data_['lengths'][idx]==n and score==data_['costs'][idx] and equal(integration,data_['final_integration'][idx])
                    queries+=1;iteration_ids.append(idx)
                    if best_id is None or (score,idx)<(best_cost,best_id):best=theta.copy();best_cost=score;best_id=idx
                    goal_found=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision'));active=None
                    if goal_found:break
                ordered=sorted(iteration_ids,key=lambda idx:(float(data_['costs'][idx]),idx));elite=ordered[:3]
                before_mean=mean.copy();before_sd=sd.copy()
                if not goal_found:
                    assert len(elite)==3;el=data_['coefficients'][elite]
                    mean=.5*mean+.5*el.mean(0);sd=np.maximum(np.asarray(p['minimum_sd']),.5*sd+.5*el.std(0))
                traces.append(dict(iteration=it,queries=iteration_ids,elite=elite,mean_before=before_mean.tolist(),sd_before=before_sd.tolist(),mean_after=mean.tolist(),sd_after=sd.tolist(),best_id=int(best_id),best_cost=float(best_cost),safe_GOAL=goal_found))
                if goal_found:break
            s.restore(snapshot);phase=f'PASS{rep}_ACTUAL_SELECTED_COMPLETE_EXECUTION';command();u=data_['controls'][best_id]
            for t in range(int(data_['lengths'][best_id])):
                row,hd=s.step(u[t],owner=1)
                assert equal(row[5],data_['next_poses'][best_id,t]) and row[2]==data_['counts'][best_id,t] and equal(np.asarray(row[6],np.uint8),data_['flags'][best_id,t])
                if rep==0:
                    if t%10==0 or s.done():persist_actual()
                else:assert len(s.rows)<=len(actualref['states']) and equal(hd,actualref['hold_sha256'][len(s.rows)-1])
                if s.done():break
            assert s.done() and equal(K.R.integration(s.env),data_['final_integration'][best_id]);a=s.mixed();result=s.result()
            if rep==0:persist_actual();actualref=copy.deepcopy(a)
            else:assert hashes(a)==hashes(actualref) and all(equal(v,actualref[key]) for key,v in a.items())
            results.append(dict(pass_index=rep,representation=p['representation'],prefix_hash=prefix_hash,prefix20_history_exact=True,queries=queries,best_id=int(best_id),goal_found=goal_found,mixed=result,iterations=traces,mixed_hashes=hashes(a),search_hashes=hashes(data_),independent_sample=rep==0))
            save(out/'passes.json',results);s.close();s=None
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if s:
            try:
                partial=dict(packet=s.packet(),integration=K.R.integration(s.env),history_poses=w.get(s.env)[0],history_goals=w.get(s.env)[1],history_times=w.get(s.env)[2],steps=np.asarray(s.steps),all_physics=np.asarray(physics))
                if active:
                    # Full candidate is bounded by380 holds; preserve its actual
                    # rows (not fabricated completion) in the reserved512KiB.
                    branch=s.arrays(actor_fields=False);write_npz(out/'partial_candidate.npz',S.packed(branch),512*1024)
                write_npz(out/'interruption.npz',partial,32*1024)
            except BaseException:save(out/'partial_error.json',dict(error=traceback.format_exc()))
        save(out/'failure.json',dict(error=error,phase=phase,active=active,physics=physics,commands=commands))
    finally:
        if s:s.close()
        if w:w.close()
        signal.alarm(0)
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,Actor=0,optimizer=0,CEM_distribution_updates=sum(not x['safe_GOAL'] for x in results[0]['iterations']) if results else None,
            passes=results,all_workers_stopped=True,teacher_safe_GOAL=bool(state=='COMPLETE' and results[0]['mixed']['success']),autonomous_six_stable=False,old275=None,originalsix54=None,fresh16=None,reload78=None,
            files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(out/'summary.json',summary);save(out/'status.json',dict(state=state,updated_epoch=time.time(),physics=physics,commands=commands));capacity(out,p['final_bytes']);print(json.dumps(dict(state=state,physics=physics,commands=commands)))
    assert state=='COMPLETE','Engineering failure stops chain; no scientific fallback bypass'

def audit(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');configure(out);base=science(p);s=read(out/'summary.json');stopped(out);assert s['state']=='COMPLETE' and not (out/'audit.json').exists()
    f,g=s['passes'];a=S.arrays(out/'mixed.npz');d=load_npz(out/'search.npz');human=S.arrays(base['demonstration'])['actions'];source=S.arrays(R.SOURCE/'centre.npz')
    assert f['queries']==g['queries'] and f['best_id']==g['best_id'] and f['iterations']==g['iterations']
    assert all(equal(a[key][:20],source[key][:20]) for key in NAMES)
    mean=np.zeros((16,2));sd=np.broadcast_to(np.asarray(p['initial_sd']),(16,2)).copy();best=None;best_id=None;best_cost=float('inf')
    for it,rec in enumerate(f['iterations']):
        proposed=candidates(mean,sd,best,it,p)
        for idx in rec['queries']:
            theta=proposed[idx%12];u=controls(theta,human,p);assert equal(theta,d['coefficients'][idx]) and equal(u,d['controls'][idx]);n=int(d['lengths'][idx]);assert n and not d['flags'][idx,:n-1].any() and d['flags'][idx,n-1].any()
            cost=objective(u,n,d['flags'][idx,n-1],d['remaining_m'][idx]);assert cost==d['costs'][idx]
            if best_id is None or (cost,idx)<(best_cost,best_id):best=theta.copy();best_id=idx;best_cost=cost
        elite=sorted(rec['queries'],key=lambda idx:(float(d['costs'][idx]),idx))[:3];assert elite==rec['elite'] and best_id==rec['best_id']
        if not rec['safe_GOAL']:
            mean=.5*mean+.5*d['coefficients'][elite].mean(0);sd=np.maximum(np.asarray(p['minimum_sd']),.5*sd+.5*d['coefficients'][elite].std(0))
        assert equal(mean,np.asarray(rec['mean_after'])) and equal(sd,np.asarray(rec['sd_after']))
    assert best_id==f['best_id'];n=int(d['lengths'][best_id]);assert len(a['states'])==20+n and equal(a['actions'][20:],d['controls'][best_id,:n]) and np.all(a['ownership'][:20]==0) and np.all(a['ownership'][20:]==1)
    physical=2*(2000+int(d['counts'].sum())+int(a['native_counts'][20:].sum()));assert physical==s['physics']<=p['max_physics'] and s['commands']==2*(2+f['queries'])<=p['max_commands']
    assert hashes(a)==f['mixed_hashes']==g['mixed_hashes'] and hashes(d)==f['search_hashes']==g['search_hashes']
    q=R.actor(base);hidden=None;goal=np.asarray(read(R.SOURCE/'cases.json')[0]['goal_xy'])
    for i,x in enumerate(a['states']):
        assert np.abs(K.H.rebuild(x,a['poses'][i],a['history_poses'][i],goal,a['history_goals'][i])-x).max()<=2e-6
        before=np.zeros(16) if hidden is None else hidden[0,0].numpy().copy()
        with torch.no_grad():mode,_,hidden,_=q.step(torch.tensor(x[None]),hidden)
        assert equal(before,a['hidden_before'][i])
        for key,v in q.tape[-1].items():assert equal(v,a['actor_proposal_'+key][i])
        if i<20:assert equal(mode[0].numpy(),a['actions'][i])
        d0=np.linalg.norm(a['poses'][i,:2]-goal);d1=np.linalg.norm(a['next_poses'][i,:2]-goal);n=int(a['native_counts'][i]);z=K.R.GAMMA**n;flags=a['end_flags'][i];term=bool(flags.any());rw=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not term)
        if term:rw+=K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
        assert abs(a['rewards'][i]-rw)<1e-12
    for path,h in s['files'].items():assert sha(out/path)==h
    save(out/'audit.json',dict(state='PASS',native=0,Actor=0,optimizer=0,all_CEM_candidates_elites_distribution_updates_reconstructed=True,actual_teacher_and_two_pass_native_arrays_exact=True,physics=physical,commands=s['commands'],teacher_safe_GOAL=s['teacher_safe_GOAL'],autonomous_six_stable=False));print('FULL_HORIZON_SEARCH_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
