"""Two fixed matched-budget early-state feedback-controller searches.

No Actor learning. Every gain proposal is charged real native simulation;
the second full pass is exact physical repetition, not independent new data.
"""
from t422_common import *
def samples(mu,sd,rng,p):
    x=rng.normal(size=(p['search_population'],6))*sd+mu;x[0]=mu
    return np.clip(x,p['gain_low'],p['gain_high'])
def main():
    E.N.offline();ban_learning();p=read(OUT/'protocol.json');verify(p);folder=OUT/'search';assert not (folder/'summary.json').exists();folder.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers()
    start=time.time();deadline=min(start+p['search_seconds'],p['native_stop_absolute']);physics=0;commands=0;state='FAILED';phase='SETUP';last=0.;s=w=None;records=[];cems=[];snapshots=[];best=None;best_array=None;result={}
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(folder/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline))
    def guard(physical=False):
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5 and physics<=p['search_physics'] and commands<=p['search_commands']
        if not physical and time.time()-last>5:
            last=time.time();verify(p);save(folder/'status.json',dict(state=phase,epoch=last,pid=os.getpid(),physics=physics,commands=commands,recorded=len(records),safe_GOAL=sum(r['success'] for r in records if r['pass_index']==0)))
    def charge():
        nonlocal physics
        physics+=1;assert physics<=p['search_physics']
    def logcommand(role,before):
        with (folder/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,role=role,physics=physics-before,pid=os.getpid()))+'\n')
    try:
        with (folder/'console.log').open('w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            native,create,w=setup(p,folder/'native');commands=1;logcommand('SETUP',physics)
            ref=R.load_reference(p['reference_path'],p['reference_sha256']);source=E.N.S.arrays(p['source_mixed']);first=p['recovery_prefixes'][0]
            for mode in p['search_modes']:
                first_arrays=[]
                for repeat in range(2):
                    phase=mode+'_PREFIX'+str(repeat);before=physics;commands+=1;q=frozen(p);s=E.L.Session(p['job'],native,create,w,guard,charge,q)
                    assert E.equal(s.initial,ref['arrays']['initial_native_integration']) and E.equal(s.packet(),ref['arrays']['states'][0])
                    for i in range(first):s.step(source['actions'][i],owner=0)
                    prefix=E.payload(s)
                    for key in ['states','actions','poses','next_poses','native_counts','hold_sha256','hidden_before','locations']:
                        assert E.equal(prefix[key],source[key][:first]),key
                    snap=s.capture();signature=snap_hash(s);snapshots.append(dict(mode=mode,pass_index=repeat,hash=signature,prefix_digest=digest(prefix),physics=s.steps));logcommand(phase,before)
                    if repeat:assert snapshots[-2]['hash']==signature
                    rng=np.random.default_rng(p['search_seed']);mu=np.asarray(p['gain_mu']);sd=np.asarray(p['gain_sd'])
                    for gen in range(p['search_generations']):
                        theta=samples(mu,sd,rng,p);generation=[]
                        for j,gains in enumerate(theta):
                            index=gen*p['search_population']+j;phase=f'{mode}_PASS{repeat}_CANDIDATE{index}';guard();assert physics+(40000-first*100)<=p['search_physics'];commands+=1;before=physics;s.restore(snap);assert snap_hash(s)==signature
                            a=finish_feedback(s,ref,first,gains,mode);row=s.result();row.update(mode=mode,pass_index=repeat,index=index,gains=gains.tolist(),physics=physics-before,total_native=int(a['native_counts'].sum()),length=len(a['actions']),score=control_score(row,a),array_digest=digest(a),array_hashes={k:digest(v) for k,v in a.items()},first=first)
                            if repeat==0:
                                first_arrays.append(a);file=folder/'compact'/f'{mode}_{index:03}.npz';file.parent.mkdir(exist_ok=True);E.N.write_npz(file,compact_episode(a),96*1024);row.update(compact_path=str(file),compact_sha256=sha(file))
                                if best is None or (row['score'],p['search_modes'].index(mode),index)<(best['score'],p['search_modes'].index(best['mode']),best['index']):best=copy.deepcopy(row);best_array=copy.deepcopy(a)
                            else:
                                prior=next(r for r in records if r['mode']==mode and r['pass_index']==0 and r['index']==index)
                                for key in a:assert E.equal(a[key],first_arrays[index][key]),(mode,index,key)
                                row.update(compact_path=prior['compact_path'],compact_sha256=prior['compact_sha256'])
                                assert {k:v for k,v in row.items() if k!='pass_index'}=={k:v for k,v in prior.items() if k!='pass_index'}
                            records.append(row);generation.append(row)
                            with (folder/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                            logcommand(phase,before)
                        elite=sorted(range(len(generation)),key=lambda k:(generation[k]['score'],k))[:p['search_elite']]
                        entry=dict(mode=mode,pass_index=repeat,generation=gen,mu=mu.tolist(),sd=sd.tolist(),theta=theta.tolist(),scores=[r['score'] for r in generation],elite=elite)
                        mu=theta[elite].mean(0);sd=np.maximum(theta[elite].std(0),p['gain_min_sd']);entry.update(next_mu=mu.tolist(),next_sd=sd.tolist());cems.append(entry);save(folder/'cem.json',cems)
                    s.close();s=None
            phase='SELECTED_FULL_ORIGIN_REPLAY';commands+=1;before=physics;assert physics+40000<=p['search_physics']
            a,row=recovery(p,native,create,w,guard,charge,p['student_start_model'],first,best['gains'],best['mode'])
            for key in a:assert E.equal(a[key],best_array[key]),('Selected full origin mismatch',key)
            path=folder/'selected.npz';E.N.write_npz(path,E.N.S.packed(a),p['query_full_bytes']);logcommand(phase,before)
            result=dict(best=best,path=str(path),sha256=sha(path),valid_recovery=bool(row['success'] and not row['collision']),Actor_updates=0,no_six_metre_acceptance=True)
            save(folder/'snapshots.json',snapshots);state='COMPLETE'
    except BaseException:
        save(folder/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=physics,commands=commands,no_restart=True))
        if s is not None and s.rows:
            try:E.N.write_npz(folder/'partial.npz',E.N.S.packed(E.payload(s)),p['query_full_bytes'])
            except BaseException:pass
    finally:
        if s:s.close()
        if w:w.close()
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=physics,commands=commands,result=result,Actor_updates=0,optimizer_steps=0,all_sessions_closed=True));save(folder/'status.json',dict(state=state,epoch=time.time(),physics=physics,commands=commands));E.N.manifest(folder);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T422_SEARCH_COMPLETE',result.get('valid_recovery'),flush=True)
if __name__=='__main__':main()
