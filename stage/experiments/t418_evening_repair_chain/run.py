"""One pre-registered offline-capable child phase, no automatic extra trials.

Entry: run.py /absolute/path/to/phase/protocol.json. The root driver is the only
finalizer/launcher. Rejected models can be retained as evidence, never deployed.
"""
from repair_core import *
import fcntl

def verify(p,protocol_file):
    for f,h in p['inputs'].items():assert sha(f)==h,f
    root_protocol=Path(p.get('root_protocol_path',HERE/'protocol.json'))
    assert sha(root_protocol)==p['root_protocol_sha256']
    assert (HERE/'protocol.json').read_bytes()==root_protocol.read_bytes()
    assert Path(protocol_file).resolve()==OUT/'protocol.json'
    assert p['phase'] in ['repair','protect','rl']
    assert p['max_rounds']<=(4 if p['phase']=='rl' else 2)
    capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=2**30
    return N.science(p)

def main(protocol_file):
    global OUT
    N.offline();p=read(protocol_file);OUT=Path(p['out']);set_out(OUT);base=verify(p,protocol_file)
    assert not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);L.install_stop_handlers()
    q,opt=learner(base,p,model=p['student_start_model']);start=time.time()
    p=dict(p,native_stop_epoch=min(start+p['main_seconds'],p['native_stop_absolute']))
    learnstop=min(start+p['learning_seconds'],p['learning_stop_absolute'])
    counts=mp.get_context('spawn').RawArray('q',5);commands=0;pool=None;last=0.;phase='SETUP'
    rows=[];reps=[];rounds=[];updates=dict(live=0,shadow=0);result={};state='FAILED';rollback=None
    reason='UNFINISHED';anyaccepted=False;rejections=0;query_sources=copy.deepcopy(p.get('query_sources',[]))
    initial_ids=set();selected=set();new_query_count=0
    def query_uid(entry):
        row,case=entry['row'],entry['case']
        return (row['policy_hash'],row['job_id'],case.get('teacher_kind','left_carry'),row['array_digest'])
    assert p['query_deduplication']=='policy_hash_job_teacher_kind_array_digest; unique labels once; all physical repeats charged'
    assert len({query_uid(entry) for entry in query_sources})==len(query_sources)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=p['native_stop_epoch'],learning_deadline=learnstop,phase=p['phase']))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<=p['max_physics'] and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();capacity(OUT,p['final_bytes']);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,rounds=len(rounds),updates=updates))
    def lg():guard();assert time.time()<learnstop-5
    def event(tag,which):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([tag,which,where,k])+'\n')
        return emit
    def evaluate(jobs,role,full=False,seeds=None,refs=None,case=None,model_path=None):
        nonlocal commands,phase
        phase=role;output=[];model=export(q,opt,OUT/'sampling.pt.gz') if model_path is None else dict(path=str(model_path),sha256=sha(model_path))
        if model_path is not None:
            stored=torch.load(io.BytesIO(gzip.decompress(Path(model_path).read_bytes())),weights_only=False,map_location='cpu')
            assert stored['hashes']==L.statehash(q,opt) and stored['policy_hash']==q.phash()
        for off in range(0,len(jobs),4):
            js=jobs[off:off+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics']
            ids=list(range(commands+1,commands+len(js)+1));commands+=len(js);assert commands<=p['max_commands']
            req=[dict(model=model['path'],job=j,first=0,seed=None if seeds is None else seeds[off+i]) for i,j in enumerate(js)]
            if case is not None:assert len(js)==1 and seeds is None;req[0].update(query=True,case=case,first=case['first'])
            for i,((a,r),job,cid) in enumerate(zip(pool.request(req),js,ids)):
                name=role+'_'+str(off+i) if seeds is not None else role+'_'+job['id']
                row=dict(r,role=name,job_id=job['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[off+i],loaded_model_sha256=model['sha256'])
                if case is not None and not case.get('source_control',False):
                    uid=query_uid(dict(row=row,case=case));prior=next((entry for entry in query_sources if query_uid(entry)==uid),None)
                    row.update(duplicate_of=None if prior is None else prior['row']['path'],unique_training_coverage=prior is None)
                if refs is None:
                    path=OUT/'episodes'/f'{len(rows):04}.npz';limit=p['query_full_bytes'] if case else p['full_bytes']
                    N.write_npz(path,N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},limit if full else 5*1024)
                    row.update(path=str(path),sha256=sha(path),full=bool(full));rows.append(row);filename='cases.jsonl'
                else:
                    for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==refs[off+i][key],key
                    row.update(exact=True,independent_sample=False);reps.append(row);filename='replays.jsonl'
                with (OUT/filename).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=name,physics=r['physics'],pid=r['pid'],error=None))+'\n')
                output.append((a,row))
        return output
    def joint(role,full=False):
        vals=[]
        for suffix,jobs in [('old8',p['old8_jobs']),('left5',p['left5_jobs']),('right',[p['right_job']])]:vals+=evaluate(jobs,role+'_'+suffix,full=full)
        actual=[r for a,r in vals];ids=sorted(r['job_id'] for r in actual if r['success'])
        check=dict(old8=tally(actual[:8]),left5=tally(actual[8:13]),right1=tally(actual[13:]),policy_hash=q.phash(),success_ids=ids,all14=len(ids)==14)
        return vals,check
    def keep(check):return initial_ids<=set(check['success_ids']) and selected<=set(check['success_ids'])
    def refresh():
        g['corrections']=[checked_query(q,entry,p) for entry in query_sources]
        save(OUT/'query_sources.json',query_sources)
    def teacher_jobs():
        # The target identities are pre-registered. Successful targets are not
        # queried again and no failed action/trajectory receives success labels.
        return [(p['old8_jobs'][6],'safe_reset'),(p['left5_jobs'][3],'left_carry')]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T418 child native deadline')))
    signal.alarm(max(1,int(p['native_stop_epoch']-time.time()-2)))
    try:
        pool=Pool(p,counts,guard);commands=4
        for i,row in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,pid=row['pid'],error=None))+'\n')
        initial_values,initial=joint('initial',full=True);initial_ids=set(initial['success_ids']);selected=initial_ids.copy();result['initial']=initial
        if q.phash()==p.get('initial_expected_policy_hash'):
            assert initial_ids==set(p['initial_expected_success_ids'])
        if p.get('initial_control_rows'):
            historical=p['initial_control_rows'];assert len(historical)==14
            for old,(_,new) in zip(historical,initial_values):
                for key in ['job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert old[key]==new[key],key
            result['initial_historical_repeat']=dict(conditions=14,new_independent_samples=0,physics_charged=sum(r['physics_steps'] for a,r in initial_values))
        g=build_groups(q,initial_values,p);refresh();reason='REGISTERED_ROUNDS_COMPLETE'
        for rnd in range(p['max_rounds']):
            if time.time()+p.get('minimum_cycle_seconds',180)>=learnstop:reason='NO_FULL_REGISTERED_LEARNING_WINDOW';break
            before=L.state(q,opt);rollback=before;folder=OUT/'blocks'/str(rnd);folder.mkdir();record=dict(round=rnd,controls=[],queries=[],bc=None,rl=None,selected_success_ids=sorted(selected))
            if p['phase']!='rl':
                source=export(q,opt,folder/'before.pt.gz')
                if rnd==0:
                    for job,kind in teacher_jobs():
                        case=dict(id='control_'+kind,job=job,first=0,teacher_kind=kind,source_control=True)
                        values=evaluate([job],'control_'+kind,full=True,case=case);a,row=values[0]
                        if row['success']:evaluate([job],'control_'+kind+'_exact',refs=[row],case=case)
                        record['controls'].append(dict(case=case,row=row,source_student_model=source['path'],source_student_sha256=source['sha256']))
                for job,kind in teacher_jobs():
                    if job['id'] in selected:continue
                    case=dict(id=f'query_{rnd}_{kind}',job=job,first=p.get('teacher_first',20),teacher_kind=kind,source_control=False)
                    values=evaluate([job],f'teacher_{rnd}_{kind}',full=True,case=case);a,row=values[0]
                    entry=dict(case=case,row=row,source_student_model=source['path'],source_student_sha256=source['sha256']);record['queries'].append(entry)
                    if row['success']:
                        assert not row['collision'] and not row['timeout'];evaluate([job],f'teacher_{rnd}_{kind}_exact',refs=[row],case=case)
                        if row['duplicate_of'] is None:query_sources.append(entry);new_query_count+=1
                        else:entry['duplicate_of']=row['duplicate_of']
                refresh()
                successful_new=[entry for entry in record['queries'] if entry['row']['success']]
                if successful_new:
                    shadow,so=learner(base,p);L.load(shadow,so,before)
                    fit=fit_bc(q,opt,g,p,lg,event('BC'+str(rnd),'live'));checkfit=fit_bc(shadow,so,g,p,lg,event('BC'+str(rnd),'shadow'))
                    assert fit==checkfit and L.statehash(q,opt)==L.statehash(shadow,so)
                    model=export(q,opt,folder/'bc_proposed.pt.gz');_,check=joint(f'BC_{rnd}_proposal')
                    accepted=keep(check) and bool(set(check['success_ids'])-initial_ids)
                    if accepted:selected=set(check['success_ids']);anyaccepted=True
                    else:L.load(q,opt,before)
                    record['bc']=dict(fit=fit,proposed_model=model,checks=check,accepted=accepted,retained_hashes=L.statehash(q,opt))
                else:record['reason']='NO_NEW_VERIFIED_TEACHER_RECOVERY_LABELS'
            # Pure-student RL is always available in its own registered phase.
            # Repair phases only enter it after an accepted repair restores all8
            # old centres, so old capability defects cannot be silently ignored.
            eligible=(p['phase']=='rl' or (record['bc'] is not None and record['bc']['accepted'] and record['bc']['checks']['old8']['success']==8))
            if eligible:
                opt=torch.optim.Adam(params(q),lr=p['lr']);rstate=L.state(q,opt);rollback=rstate
                job=p['left5_jobs'][3];seeds=[[p['seed'],p.get('phase_seed_offset',0),rnd,k] for k in range(4)]
                vals=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,refs=[r for a,r in vals])
                shadow,so=learner(base,p);L.load(shadow,so,rstate)
                fit=fit_ppo(q,opt,[a for a,r in vals],g,p,rnd,lg,event(rnd,'live'));checkfit=fit_ppo(shadow,so,[a for a,r in vals],g,p,rnd,lg,event(rnd,'shadow'))
                assert fit==checkfit and L.statehash(q,opt)==L.statehash(shadow,so)
                model=export(q,opt,folder/'rl_proposed.pt.gz');_,check=joint(f'RL_{rnd}_proposal');accepted=fit['numeric_accept'] and keep(check)
                if accepted:selected=set(check['success_ids']);anyaccepted=True
                else:L.load(q,opt,rstate)
                record['rl']=dict(fit=fit,proposed_model=model,checks=check,accepted=accepted,retained_hashes=L.statehash(q,opt),sampling=tally([r for a,r in vals]),job=job,seeds=seeds)
            round_accepted=bool((record['bc'] and record['bc']['accepted']) or (record['rl'] and record['rl']['accepted']))
            rejections=0 if round_accepted else rejections+1;record.update(selected_success_ids=sorted(selected),consecutive_rejections=rejections)
            rounds.append(record);save(folder/'result.json',record);save(OUT/'rounds.json',rounds);rollback=None
            if rejections>=2:reason='TWO_CONSECUTIVE_REJECTIONS';break
            if p['phase']!='rl' and not any(entry['row']['success'] for entry in record['queries']):reason='BOTH_REGISTERED_TEACHERS_NO_NEW_RECOVERY';break
        result.update(model=export(q,opt,OUT/'candidate.pt.gz'),any_learning_accepted=anyaccepted,new_verified_query_count=new_query_count,main54=None,old275=None,fresh16=None,reload78=None,stable_wide_six=False)
        _,final=joint('final_joint');result['final_joint']=final;assert initial_ids<=set(final['success_ids']) and selected<=set(final['success_ids'])
        centres=evaluate(p['centres'],'final_centres');result['main_centres']=tally([r for a,r in centres]);evaluate(p['centres'],'final_centres_exact',refs=[r for a,r in centres],model_path=result['model']['path'])
        if final['all14']:
            dev=[r for a,r in evaluate(p['main_jobs'],'main54')];old=[r for a,r in evaluate(base['final_old_jobs'],'old275')];result.update(main54=tally(dev),old275=tally(old))
            if all(row['success'] for row in [*dev,*old]):
                for f,h in p['fresh_ledger'].items():assert sha(f)==h
                fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
                if all(r['success'] for r in fresh):
                    look={r['job_id']:r for r in [*dev,*old,*fresh]};reload=evaluate(p['reload_jobs'],'reload78',refs=[look[j['id']] for j in p['reload_jobs']],model_path=result['model']['path'])
                    result.update(reload78=dict(exact=len(reload),independent_sample=False),stable_wide_six=len(reload)==78)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();exact=None
        if rollback is not None:L.load(q,opt,rollback);exact=digest(L.state(q,opt))==digest(rollback);export(q,opt,OUT/'interrupted_retained.pt.gz')
        save(OUT/'failure.json',dict(error=error,phase=phase,physics=sum(counts),commands=commands,updates=updates,interrupted_Actor_Adam_rollback=exact))
    finally:
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        summary=dict(state=state,pid=os.getpid(),phase=p['phase'],elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=updates,rounds=rounds,result=result,reason=reason,
            query_sources=query_sources,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False,narrow_paused=True)
        save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=updates));N.manifest(OUT)
    assert state=='COMPLETE'

if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
