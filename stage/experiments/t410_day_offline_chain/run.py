"""A/B parameter isolation and projected learning; C verified correction/PPO.

Each directory is written exactly once. The registered supervisor selects only
audited retained models; all autonomous evaluations use the sole Actor.
"""
from day_core import *
import fcntl

def configuration(folder):
    global OUT
    path=Path(folder)
    if path.is_absolute():assert path.parent==ROOT_OUT;folder=path.name
    root=read(ROOT_OUT/'protocol.json');assert (HERE/'protocol.json').read_bytes()==(ROOT_OUT/'protocol.json').read_bytes()
    for f,h in root['inputs'].items():assert sha(f)==h,f
    spec=next(s for s in root['stages'] if s['folder']==folder);out=ROOT_OUT/folder;runtime=read(out/'runtime.json');assert runtime['initial_model_sha256']==sha(runtime['initial_model'])
    p=dict(root['common']);p.update(spec);p.update(runtime);p['inputs']=root['inputs'];p['root']=root;OUT=out;set_out(out);return p,N.science(p)

def main(folder):
    N.offline();p,base=configuration(folder);assert not (OUT/'summary.json').exists();L.install_stop_handlers();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q,opt=learner(base,p);g=groups(q,[N.S.arrays(f) for f in p['old_history_files']],p,N.old_data(q,base,lambda:None));start=time.time();counts=mp.get_context('spawn').RawArray('q',5)
    commands=0;pool=None;last=0.;phase='SETUP';cases=[];replays=[];blocks=[];updates=dict(live=0,shadow=0);result={};state='FAILED';selected=[];reason='UNFINISHED';rollback_state=None
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,learning_deadline=p['learning_stop_epoch'],native_deadline=p['native_stop_epoch'],arm=p['id']))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<=p['max_physics'] and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,blocks=len(blocks),updates=updates,arm=p['id']))
    def learn_guard():guard();assert time.time()<p['learning_stop_epoch']-5
    def block_admission():
        # Every new recovery block reserves complete worst-case evidence before
        # any native advance. The fixed byte budget can truncate the max16 list.
        own=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
        return own+9*p['full_bytes']+2*512*1024+256*1024<=p['final_bytes'] and time.time()+p['minimum_cycle_seconds']<p['learning_stop_epoch']
    def evaluate(jobs,role,full=False,seeds=None,refs=None,controls=None,force_first=None,keep_teacher=False,model_path=None):
        nonlocal commands,phase
        phase=role;out=[];model=export(q,opt,OUT/'sampling.pt.gz') if model_path is None else dict(path=str(model_path),sha256=sha(model_path))
        if model_path:
            stored=torch.load(io.BytesIO(gzip.decompress(Path(model_path).read_bytes())),weights_only=False,map_location='cpu');assert stored['hashes']==L.statehash(q,opt) and stored['policy_hash']==q.phash()
        for off in range(0,len(jobs),4):
            js=jobs[off:off+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics'];ids=list(range(commands+1,commands+len(js)+1));commands+=len(js);assert commands<=p['max_commands']
            requests=[dict(model=model['path'],job=j,first=0,seed=None if seeds is None else seeds[off+i]) for i,j in enumerate(js)]
            if controls is not None:
                assert len(js)==1 and seeds is None;requests[0].update(controls=controls,force_first=force_first)
            vals=pool.request(requests)
            for i,((a,r),j,cid) in enumerate(zip(vals,js,ids)):
                name=role+'_'+str(off+i) if seeds is not None else role+'_'+j['id'];row=dict(r,role=name,job_id=j['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[off+i],loaded_model_sha256=model['sha256'],teacher=controls is not None,force_first=force_first)
                if refs is None:
                    isfull=bool(full or (keep_teacher and row['success']));path=OUT/'episodes'/f'{len(cases):04}.npz';N.write_npz(path,N.S.packed(a) if isfull else {k:a[k] for k in ['actions','native_counts','end_flags']},p['full_bytes'] if isfull else 5*1024);row.update(path=str(path),sha256=sha(path),full=isfull);cases.append(row);filename='cases.jsonl'
                else:
                    ref=refs[off+i]
                    for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[k]==ref[k],k
                    row.update(exact=True,independent_sample=False);replays.append(row);filename='replays.jsonl'
                with (OUT/filename).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=name,physics=r['physics'],pid=r['pid'],error=None))+'\n')
                out.append((a,row))
        return out
    def joint(role):
        rows=[r for a,r in evaluate([*p['old8_jobs'],*p['left5_jobs'],p['right_job']],role)];ids=sorted(r['job_id'] for r in rows if r['success']);return dict(policy_hash=q.phash(),success_ids=ids,all14=len(ids)==14,old8=tally(rows[:8]),left5=tally(rows[8:13]),right1=tally(rows[13:]))
    mandatory={j['id'] for j in p['old8_jobs']}|{p['right_job']['id']}
    def acceptable(check):return mandatory<=set(check['success_ids']) and set(selected)<=set(check['success_ids'])
    def event(tag,which):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([tag,which,where,k])+'\n')
        return emit
    def record(tag,before,fit,check,model,keep,extra=None):
        nonlocal selected,rollback_state
        if not keep:L.load(q,opt,before)
        else:selected=check['success_ids']
        b=dict(tag=tag,fit=fit,checks=check,proposed_model=model,accepted=bool(keep),selected_success_ids=selected.copy(),retained_hashes=L.statehash(q,opt),**(extra or {}));blocks.append(b);folder=OUT/'blocks'/str(tag);folder.mkdir(exist_ok=True);save(folder/'result.json',b);save(OUT/'blocks.json',blocks)
        rollback_state=None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('registered native hard deadline')));signal.alarm(max(1,int(p['native_stop_epoch']-time.time()-2)))
    try:
        pool=Pool(p,counts,guard);commands=4
        for i,r in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,pid=r['pid'],error=None))+'\n')
        initial=joint('initial');selected=initial['success_ids'];assert mandatory<=set(selected);result['initial']=initial
        if p['kind'] in ['lateral','projected']:
            before=L.state(q,opt);rollback_state=before;shadow,so=learner(base,p);L.load(shadow,so,before);fn=fit_A if p['kind']=='lateral' else fit_B
            live=fn(q,opt,g,p,learn_guard,event('warm','live'));other=fn(shadow,so,g,p,learn_guard,event('warm','shadow'));assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);model=export(q,opt,OUT/'warm_proposed.pt.gz');check=joint('warm_proposal');keep=acceptable(check);record('warm',before,live,check,model,keep)
            reason='WARM_ACCEPTED' if keep else 'WARM_REJECTED'
            if keep:
                opt=torch.optim.Adam(params(q),lr=p['lr']);result['fresh_Adam_after_warm']=True
                for rnd,job in enumerate(p['rl_jobs']):
                    if not block_admission():reason='BLOCK_ADMISSION_STOP';break
                    before=L.state(q,opt);rollback_state=before;seeds=[[p['seed'],rnd,k] for k in range(4)];vals=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,refs=[r for a,r in vals]);batch=[a for a,r in vals];shadow,so=learner(base,p);L.load(shadow,so,before)
                    live=ppo_fit(q,opt,batch,g,p,rnd,learn_guard,event(rnd,'live'),projected=p['kind']=='projected');other=ppo_fit(shadow,so,batch,g,p,rnd,learn_guard,event(rnd,'shadow'),projected=p['kind']=='projected');assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);path=OUT/'blocks'/str(rnd);path.mkdir(exist_ok=True);model=export(q,opt,path/'proposed.pt.gz');check=joint(f'proposal_{rnd}');keep=live['numeric_accept'] and acceptable(check);record(rnd,before,live,check,model,keep,dict(job=job,sampling=tally([r for a,r in vals]),seeds=seeds));reason='BOUNDED_TWO_SIDE_RL_COMPLETE'
        elif p['kind']=='recovery':
            reason='MAX_CYCLES_COMPLETE';consecutive_rejects=0
            for rnd in range(p['cycles']):
                if not block_admission():reason='BLOCK_STORAGE_OR_LEARNING_TIME_ADMISSION_STOP';break
                # A fixed missing-LEFT-first rule uses only registered development
                # results. Teachers do not select candidates or consume holdout.
                missing=[i for i,j in enumerate(p['left5_jobs']) if j['id'] not in selected];idx=missing[0] if missing else rnd%5;job=p['left5_jobs'][idx];reference=N.S.arrays(p['left_history_files'][idx]);teacher=None;queries=[];before=L.state(q,opt);rollback_state=before
                for variant in p['teacher_advance_holds']:
                    ids=np.minimum(np.arange(400)+variant,len(reference['actions'])-1);controls=reference['actions'][ids].copy();vals=evaluate([job],f'teacher_{rnd}_{variant}',controls=controls,force_first=p['teacher_first'],keep_teacher=True);a,row=vals[0];queries.append(dict(variant=variant,row=row))
                    if row['success']:
                        evaluate([job],f'teacher_{rnd}_{variant}_exact',controls=controls,force_first=p['teacher_first'],refs=[row]);teacher=(a,row);break
                corrections=[]
                if teacher:
                    a,row=teacher;x=torch.tensor(a['states']);corrections=[dict(x=x,b=q.base_actions(x),target=torch.tensor(a['actions'],dtype=torch.float64),first=p['teacher_first'])]
                g['corrections']=corrections;shadow,so=learner(base,p);L.load(shadow,so,before);warm=None
                if corrections:
                    warm=fit_B(q,opt,g,p,learn_guard,event('cw'+str(rnd),'live'),steps=p['correction_steps']);other=fit_B(shadow,so,g,p,learn_guard,event('cw'+str(rnd),'shadow'),steps=p['correction_steps']);assert warm==other and L.statehash(q,opt)==L.statehash(shadow,so)
                # Fresh on-policy rollouts occur after the current finite warmup.
                seeds=[[p['seed'],rnd,k] for k in range(4)];vals=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,refs=[r for a,r in vals]);batch=[a for a,r in vals];L.load(shadow,so,L.state(q,opt));live=ppo_fit(q,opt,batch,g,p,rnd,learn_guard,event(rnd,'live'),projected=True);other=ppo_fit(shadow,so,batch,g,p,rnd,learn_guard,event(rnd,'shadow'),projected=True);assert live==other and L.statehash(q,opt)==L.statehash(shadow,so);path=OUT/'blocks'/str(rnd);path.mkdir(exist_ok=True);model=export(q,opt,path/'proposed.pt.gz');check=joint(f'proposal_{rnd}');keep=live['numeric_accept'] and acceptable(check);record(rnd,before,live,check,model,keep,dict(job=job,teacher_queries=queries,teacher_selected=None if teacher is None else teacher[1],warm_fit=warm,sampling=tally([r for a,r in vals]),seeds=seeds));consecutive_rejects=0 if keep else consecutive_rejects+1
                if consecutive_rejects>=p['reject_stop']:reason='CONSECUTIVE_RETENTION_REJECTIONS_STOP';break
        elif p['kind']=='validation':
            result['diagnostic_only_not_new_Actor_training']=True;dev=[r for a,r in evaluate(p['main_jobs'],'main54')];old=[r for a,r in evaluate(base['final_old_jobs'],'old275')];result.update(main54=tally(dev),old275=tally(old),fresh16=None,reload78=None,stable_wide_six=False);reason='FINAL_DIAGNOSTIC_COMPLETE'
            if all(r['success'] for r in [*dev,*old]):
                for f,h in p['fresh_ledger'].items():assert sha(f)==h
                fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
                if all(r['success'] for r in fresh):
                    model=export(q,opt,OUT/'candidate.pt.gz');look={r['job_id']:r for r in [*dev,*old,*fresh]};reload=evaluate(p['reload_jobs'],'reload78',refs=[look[j['id']] for j in p['reload_jobs']],model_path=model['path']);result.update(reload78=dict(exact=len(reload),independent_sample=False),stable_wide_six=len(reload)==78)
        else:raise AssertionError(p['kind'])
        # Always save a new stage-local artifact of the retained policy. Rejected
        # proposal files remain available, but never enter the next initialization.
        result['final_joint']=joint('final_joint');assert set(selected)<=set(result['final_joint']['success_ids']);selected=result['final_joint']['success_ids']
        if p['kind']=='validation' and (OUT/'candidate.pt.gz').exists():
            # Keep the exact artifact already used by the native reload gate.
            assert sha(model['path'])==model['sha256'];stored=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert stored['hashes']==L.statehash(q,opt) and stored['policy_hash']==q.phash()
        else:model=export(q,opt,OUT/'candidate.pt.gz')
        result['model']=model;state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();rollback_exact=None
        if rollback_state is not None:
            L.load(q,opt,rollback_state);rollback_exact=digest(L.state(q,opt))==digest(rollback_state);export(q,opt,OUT/'interrupted_retained.pt.gz')
        save(OUT/'failure.json',dict(error=error,phase=phase,physics=sum(counts),commands=commands,updates=updates,whole_Actor_Adam_interrupted_rollback=rollback_exact));reason='SHARED_CONTRACT_FAILURE'
    finally:
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();selected_model=result.get('model');summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=updates,blocks=blocks,result=result,reason=reason,selected_model=selected_model,selected_success_ids=selected,selection_score=[len(set(selected)&{j['id'] for j in p['left5_jobs']}),len(set(selected)&{j['id'] for j in p['old8_jobs']}),int(p['right_job']['id'] in selected)],selected_joint_all=len(selected)==14,all_workers_stopped=True,all_native_sessions_closed=True,stable_wide_six=result.get('stable_wide_six',False),seven_eight_entered=False,narrow_paused=True);save(OUT/'summary.json',summary);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=updates));N.manifest(OUT)
    assert state=='COMPLETE'

if __name__=='__main__':main(sys.argv[1])
