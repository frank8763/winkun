"""Conditionally learn only natively verified feedback recoveries, offline."""
from t422_common import *
import learn as LEARN
import multiprocessing as mp

def main():
    E.N.offline();p=read(OUT/'protocol.json');verify(p);folder=OUT/'learning';folder.mkdir(exist_ok=True)
    assert not (folder/'summary.json').exists();search=read(OUT/'search/summary.json')
    assert search['state']=='COMPLETE' and search['result']['valid_recovery'] and read(OUT/'search/audit.json')['state']=='PASS'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers();E.set_out(folder)
    base=E.N.science(p);q,opt=E.learner(base,p);start=time.time();deadline=min(start+p['learning_main_seconds'],p['native_stop_absolute']);learnstop=min(start+p['learning_seconds'],p['learning_stop_absolute'])
    config=dict(p,out=str(folder),native_stop_epoch=deadline,max_physics=p['learning_physics'])
    counts=mp.get_context('spawn').RawArray('q',5);commands=0;pool=w=None;phase='SETUP';last=0.;rows=[];reps=[];rounds=[];updates=dict(live=0,shadow=0);rollback=None;state='FAILED';reason='UNFINISHED';result={};recoveries=[];selected=set();initial_ids=set();consecutive=0
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(folder/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=deadline,learning_deadline=learnstop))
    def guard(physical=False):
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5 and sum(counts)<=p['learning_physics'] and commands<=p['learning_commands']
        if not physical and time.time()-last>5:
            last=time.time();verify(p);save(folder/'status.json',dict(state=phase,pid=os.getpid(),epoch=last,physics=sum(counts),commands=commands,updates=updates,rounds=len(rounds)))
    def lg():guard();assert time.time()<learnstop-5
    def charge():counts[0]+=1;assert sum(counts)<=p['learning_physics']
    def logcmd(cid,role,steps,pid):
        with (folder/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=role,physics=steps,pid=pid))+'\n')
    def event(tag,which,model,optimizer):
        def record(where,k):
            if where=='AFTER':updates[which]+=1
            with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps(dict(tag=tag,which=which,where=where,k=k,hashes=E.L.statehash(model,optimizer)))+'\n')
        return record
    def evaluate(jobs,role,full=False,seeds=None,refs=None,model=None):
        nonlocal phase,commands
        phase=role;sample=E.export(q,opt,folder/'sampling.pt.gz') if model is None else model;output=[]
        if model is not None:
            stored=torch.load(E.io.BytesIO(E.gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert stored['hashes']==E.L.statehash(q,opt) and stored['policy_hash']==q.phash()
        for off in range(0,len(jobs),4):
            js=jobs[off:off+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['learning_physics']
            ids=list(range(commands+1,commands+len(js)+1));commands+=len(js)
            requests=[dict(model=sample['path'],job=j,first=0,seed=None if seeds is None else seeds[off+i]) for i,j in enumerate(js)]
            for i,((a,row),job,cid) in enumerate(zip(pool.request(requests),js,ids)):
                row=dict(row,role=role,job_id=job['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[off+i],loaded_model_sha256=sample['sha256'])
                if refs is None:
                    path=folder/'episodes'/f'{len(rows):04}.npz';path.parent.mkdir(exist_ok=True)
                    E.N.write_npz(path,E.N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if full else 5*1024)
                    row.update(path=str(path),sha256=sha(path),full=full);rows.append(row);target='cases.jsonl'
                else:
                    old,prior=refs[off+i]
                    for key in a:assert E.equal(a[key],old[key]),('Independent native replay differs',role,job['id'],key)
                    for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==prior[key],key
                    row.update(exact=True,full_array_exact=True,independent_sample=False,reference_path=prior['path']);reps.append(row);target='replays.jsonl'
                with (folder/target).open('a') as f:f.write(json.dumps(row)+'\n')
                logcmd(cid,role,row['physics'],row['pid']);output.append((a,row))
        return output
    def joint(role,full=False):
        vals=evaluate(p['old8_jobs'],role+'_old8',full)+evaluate(p['left5_jobs'],role+'_left5',full)+evaluate([p['right_job']],role+'_right',full)
        actual=[r for a,r in vals];ids=sorted(r['job_id'] for r in actual if r['success'])
        return vals,dict(old8=E.tally(actual[:8]),left5=E.tally(actual[8:13]),right1=E.tally(actual[13:]),success_ids=ids,all14=len(ids)==14,policy_hash=q.phash())
    def keep(check):return initial_ids<=set(check['success_ids']) and selected<=set(check['success_ids'])
    def teacher(first,rnd,model):
        nonlocal commands,phase
        phase=f'FEEDBACK_RECOVERY_{rnd}_PREFIX{first}';before=sum(counts);commands+=1;assert before+40000<=p['learning_physics']
        a,row=recovery(p,native,create,w,guard,charge,model['path'],first,search['result']['best']['gains'],search['result']['best']['mode'],folder/f'teacher_partial_{rnd}.npz')
        row.update(role=phase,job_id=p['job']['id'],physics_steps=row['physics'],source_model=model['path'],source_model_sha256=model['sha256'])
        path=folder/'recoveries'/f'{rnd:02}.npz';path.parent.mkdir(exist_ok=True);E.N.write_npz(path,E.N.S.packed(a),p['query_full_bytes']);row.update(path=str(path),sha256=sha(path));save(folder/'recoveries'/f'{rnd:02}.json',row);logcmd(commands,phase,sum(counts)-before,os.getpid())
        entry=dict(path=str(path),sha256=sha(path),first=first,source_id=phase,row=row)
        if row['success'] and not row['collision']:
            phase+='__EXACT';before=sum(counts);commands+=1;assert before+40000<=p['learning_physics']
            again,repeat=recovery(p,native,create,w,guard,charge,model['path'],first,search['result']['best']['gains'],search['result']['best']['mode'])
            for key in a:assert E.equal(a[key],again[key]),key
            assert row['array_digest']==repeat['array_digest'];logcmd(commands,phase,sum(counts)-before,os.getpid());entry['exact']=True
            save(folder/'recoveries'/f'{rnd:02}_exact.json',dict(array_digest=repeat['array_digest'],physics=repeat['physics'],exact=True,independent_sample=False))
        return a,entry
    try:
        with (folder/'direct_console.log').open('w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            native,create,w=setup(p,folder/'direct_native');commands=1;logcmd(1,'DIRECT_SETUP',0,os.getpid())
        E.set_out(folder);pool=E.Pool(config,counts,guard);commands+=4
        for cid,row in enumerate(pool.ready,2):logcmd(cid,'WORKER_SETUP',0,row['pid'])
        initial_values,initial=joint('initial',True);initial_ids=set(initial['success_ids']);selected=initial_ids.copy();result['initial']=initial
        assert q.phash()==p['initial_expected_policy_hash'] and initial_ids==set(p['initial_expected_success_ids'])
        initial_manifest=[row for a,row in initial_values];save(folder/'initial_arrays.json',initial_manifest)
        reason='REGISTERED_ITERATIONS_COMPLETE'
        for rnd in range(p['learning_rounds']):
            if time.time()+p['minimum_cycle_seconds']>=learnstop:reason='NO_COMPLETE_LEARNING_WINDOW';break
            block=folder/'blocks'/str(rnd);block.mkdir(parents=True);before=E.L.state(q,opt);rollback=before;before_model=E.export(q,opt,block/'before.pt.gz');first=p['recovery_prefixes'][rnd%len(p['recovery_prefixes'])]
            if rnd==0:
                a=E.N.S.arrays(search['result']['path']);entry=dict(path=search['result']['path'],sha256=search['result']['sha256'],first=first,source_id='SEARCH_VERIFIED_REUSE',row=search['result']['best'],exact=True,historical_search_reuse=True,new_native=0)
            else:a,entry=teacher(first,rnd,before_model)
            record=dict(round=rnd,before_model=before_model,query=entry,bc=None,rl=None,initial_success_ids=sorted(initial_ids),selected_before=sorted(selected))
            if bool(a['end_flags'][-1,1]) and not a['end_flags'][:,0].any():
                uid=(q.phash(),first,digest(a));old=next((x for x in recoveries if tuple(x['uid'])==uid),None)
                if old is None:entry['uid']=list(uid);recoveries.append(entry)
                else:entry['duplicate_of']=old['path']
                group_entries=[dict(arrays=E.N.S.arrays(x['path']),first=x['first'],source_id=x['source_id']) for x in recoveries]
                g=LEARN.build_groups(q,initial_values,p,group_entries,p['query_sources']);record['recovery_sources']=copy.deepcopy(recoveries)
                shadow,so=E.learner(base,p);E.L.load(shadow,so,before)
                fit=LEARN.fit_bc(q,opt,g,p,lg,event('BC'+str(rnd),'live',q,opt),steps=p['bc_steps'])
                sf=LEARN.fit_bc(shadow,so,g,p,lg,event('BC'+str(rnd),'shadow',shadow,so),steps=p['bc_steps'])
                assert fit==sf and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
                model=E.export(q,opt,block/'bc_proposed.pt.gz');_,check=joint(f'BC{rnd}_proposal');accepted=keep(check) and bool(set(check['success_ids'])-selected)
                if accepted:selected=set(check['success_ids'])
                else:E.L.load(q,opt,before)
                record['bc']=dict(fit=fit,proposed_model=model,accepted=accepted,checks=check,retained_hashes=E.L.statehash(q,opt))
                if accepted and check['old8']['success']==8:
                    opt=torch.optim.Adam(E.params(q),lr=p['lr']);rstate=E.L.state(q,opt);rollback=rstate;rmodel=E.export(q,opt,block/'rl_before.pt.gz')
                    seeds=[[p['seed'],rnd,k] for k in range(4)];values=evaluate([p['job']]*4,f'train{rnd}',True,seeds);evaluate([p['job']]*4,f'train{rnd}_exact',seeds=seeds,refs=values)
                    shadow,so=E.learner(base,p);E.L.load(shadow,so,rstate)
                    fit=LEARN.fit_ppo(q,opt,[a for a,row in values],g,p,rnd,lg,event('RL'+str(rnd),'live',q,opt));sf=LEARN.fit_ppo(shadow,so,[a for a,row in values],g,p,rnd,lg,event('RL'+str(rnd),'shadow',shadow,so))
                    assert fit==sf and E.L.statehash(q,opt)==E.L.statehash(shadow,so)
                    model=E.export(q,opt,block/'rl_proposed.pt.gz');_,check=joint(f'RL{rnd}_proposal');accepted=fit['numeric_accept'] and keep(check)
                    if accepted:selected=set(check['success_ids'])
                    else:E.L.load(q,opt,rstate)
                    record['rl']=dict(before_model=rmodel,fit=fit,items=[r for a,r in values],proposed_model=model,accepted=accepted,checks=check,retained_hashes=E.L.statehash(q,opt))
            else:record['reason']='FEEDBACK_FAILED_REAL_GOAL_NO_NEW_LABEL'
            accepted=bool((record['bc'] and record['bc']['accepted']) or (record['rl'] and record['rl']['accepted']));consecutive=0 if accepted else consecutive+1
            record.update(selected_after=sorted(selected),consecutive_rejections=consecutive);rounds.append(record);save(block/'result.json',record);save(folder/'rounds.json',rounds);rollback=None
            if consecutive>=2:reason='TWO_CONSECUTIVE_REJECTIONS';break
        _,final=joint('final');result['final_joint']=final;assert initial_ids<=set(final['success_ids']) and selected<=set(final['success_ids'])
        result.update(model=E.export(q,opt,folder/'candidate.pt.gz'),old275=None,wide54=None,fresh16=None,reload78=None,stable_wide_six=False)
        centres=evaluate(p['centres'],'centres');evaluate(p['centres'],'centres_exact',refs=centres,model=result['model']);result['centres']=E.tally([r for a,r in centres])
        if final['all14']:
            dev=evaluate(p['main_jobs'],'wide54');old=evaluate(base['final_old_jobs'],'old275');result.update(wide54=E.tally([r for a,r in dev]),old275=E.tally([r for a,r in old]))
            if all(r['success'] for a,r in [*dev,*old]):
                for f,h in p['fresh_ledger'].items():assert sha(f)==h
                fresh=evaluate(p['fresh_jobs'],'fresh16');result['fresh16']=E.tally([r for a,r in fresh])
                if all(r['success'] for a,r in fresh):
                    refs={r['job_id']:(a,r) for a,r in [*dev,*old,*fresh]};reload=evaluate(p['reload_jobs'],'reload78',refs=[refs[j['id']] for j in p['reload_jobs']],model=result['model']);result.update(reload78=dict(exact=len(reload),independent_sample=False,all_arrays_native_exact=True),stable_wide_six=len(reload)==78)
        state='COMPLETE'
    except BaseException:
        if rollback is not None:E.L.load(q,opt,rollback);E.export(q,opt,folder/'interrupted_retained.pt.gz')
        save(folder/'failure.json',dict(error=traceback.format_exc(),phase=phase,physics=sum(counts),commands=commands,updates=updates,interrupted_rollback=rollback is not None,no_restart=True))
    finally:
        if pool:pool.close()
        if w:w.close()
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=updates,rounds=rounds,result=result,reason=reason,all_workers_stopped=True,all_sessions_closed=True,seven_eight_entered=False));save(folder/'status.json',dict(state=state,epoch=time.time(),physics=sum(counts),commands=commands,updates=updates));E.N.manifest(folder);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T422_LEARNING_COMPLETE',result.get('stable_wide_six'),flush=True)
if __name__=='__main__':main()
