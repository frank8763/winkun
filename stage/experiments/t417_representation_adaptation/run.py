"""One fixed representation-adaptation BC/PPO cycle, historical residual-only comparison."""
from t417_learning import *
import fcntl

def plan():
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes();return read(OUT/'protocol.json')
def verify(p):
    for f,h in p['inputs'].items():assert sha(f)==h,f
    N.capacity(OUT,p['final_bytes']);assert N.K.S.used(ROOT/'stage/runs')-N.K.S.used(OUT)+p['final_bytes']<=2**30;return N.science(p)
def main():
    N.offline();p=plan();base=verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists();set_out(OUT);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);L.install_stop_handlers()
    q,opt=learner(base,p);start=time.time();p=dict(p,out=str(OUT),native_stop_epoch=min(start+p['main_seconds'],p['native_stop_absolute']));learnstop=min(start+p['learning_seconds'],p['learning_stop_absolute']);counts=mp.get_context('spawn').RawArray('q',5);commands=0;pool=None;last=0.;phase='SETUP';rows=[];reps=[];rounds=[];updates=dict(live=0,shadow=0);result={};state='FAILED';selected=[];rollback=None;reason='UNFINISHED';anyaccepted=False
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,native_deadline=p['native_stop_epoch'],learning_deadline=learnstop))
    def guard(physical=False):
        nonlocal last
        L.check_stop();assert time.time()<p['native_stop_epoch']-5 and sum(counts)<=p['max_physics'] and commands<=p['max_commands']
        if not physical and time.time()-last>3:
            last=time.time();N.capacity(OUT,p['final_bytes']);save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,physics=sum(counts),commands=commands,rounds=len(rounds),updates=updates))
    def lg():guard();assert time.time()<learnstop-5
    def event(tag,which):
        def emit(where,k):
            if where=='AFTER':updates[which]+=1
            with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([tag,which,where,k])+'\n')
        return emit
    def evaluate(jobs,role,full=False,seeds=None,refs=None,case=None,model_path=None):
        nonlocal commands,phase
        phase=role;out=[];model=export(q,opt,OUT/'sampling.pt.gz') if model_path is None else dict(path=str(model_path),sha256=sha(model_path))
        if model_path is not None:
            s=torch.load(io.BytesIO(gzip.decompress(Path(model_path).read_bytes())),weights_only=False,map_location='cpu');assert s['hashes']==L.statehash(q,opt) and s['policy_hash']==q.phash()
        for off in range(0,len(jobs),4):
            js=jobs[off:off+4];guard();assert sum(counts)+sum(j['case']['step_cap'] for j in js)<=p['max_physics'];ids=list(range(commands+1,commands+len(js)+1));commands+=len(js);assert commands<=p['max_commands'];req=[dict(model=model['path'],job=j,first=0,seed=None if seeds is None else seeds[off+i]) for i,j in enumerate(js)]
            if case is not None:assert len(js)==1 and seeds is None;req[0].update(query=True,case=case,first=case['first'])
            for i,((a,r),j,cid) in enumerate(zip(pool.request(req),js,ids)):
                name=role+'_'+str(off+i) if seeds is not None else role+'_'+j['id'];row=dict(r,role=name,job_id=j['id'],physics_steps=int(a['native_counts'].sum()),seed=None if seeds is None else seeds[off+i],loaded_model_sha256=model['sha256'])
                if refs is None:
                    path=OUT/'episodes'/f'{len(rows):04}.npz';limit=p['query_full_bytes'] if case else p['full_bytes'];N.write_npz(path,N.S.packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']},limit if full else 5*1024);row.update(path=str(path),sha256=sha(path),full=bool(full));rows.append(row);filename='cases.jsonl'
                else:
                    for k in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[k]==refs[off+i][k],k
                    row.update(exact=True,independent_sample=False);reps.append(row);filename='replays.jsonl'
                with (OUT/filename).open('a') as f:f.write(json.dumps(row)+'\n')
                with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=cid,role=name,physics=r['physics'],pid=r['pid'],error=None))+'\n')
                out.append((a,row))
        return out
    def joint(role):
        old=[r for a,r in evaluate(p['old8_jobs'],role+'_old8')];left=[r for a,r in evaluate(p['left5_jobs'],role+'_left5')];right=[r for a,r in evaluate([p['right_job']],role+'_right')];rs=[*old,*left,*right];ids=sorted(r['job_id'] for r in rs if r['success']);return dict(old8=tally(old),left5=tally(left),right1=tally(right),policy_hash=q.phash(),success_ids=ids,all14=len(ids)==14)
    mandatory={j['id'] for j in p['old8_jobs']}|{p['right_job']['id']}
    def keep(check):return mandatory<=set(check['success_ids']) and set(selected)<=set(check['success_ids'])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T417 native deadline')));signal.alarm(max(1,int(p['native_stop_epoch']-time.time()-2)))
    try:
        pool=Pool(p,counts,guard);commands=4
        for i,r in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,pid=r['pid'],error=None))+'\n')
        initial=evaluate(p['old8_jobs'],'initial_old8',full=True);assert all(r['success'] for a,r in initial);left=[r for a,r in evaluate(p['left5_jobs'],'initial_left5')];right=[r for a,r in evaluate([p['right_job']],'initial_right')];assert all(r['success'] for r in right);selected=sorted(r['job_id'] for r in [*[r for a,r in initial],*left,*right] if r['success']);result['initial']=dict(old8=tally([r for a,r in initial]),left5=tally(left),right1=tally(right),policy_hash=q.phash(),success_ids=selected.copy(),all14=len(selected)==14)
        origin_rows=[json.loads(line) for line in Path(p['historical_control']['cases']).read_text().splitlines()][:14]
        prefix_rows=rows.copy();assert len(prefix_rows)==len(origin_rows)==14
        for oldrow,newrow in zip(origin_rows,prefix_rows):
            for key in ['role','job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert oldrow[key]==newrow[key],('Recovery prefix repeat differs',key)
        result['initial_repeat_of_T414']=dict(conditions=14,independent_new_samples=0,physics_charged_again=sum(r['physics_steps'] for r in prefix_rows))
        g=groups(q,[a for a,r in initial],p,N.old_data(q,base,lambda:None));reason='ONE_REGISTERED_REPRESENTATION_CYCLE_COMPLETE'
        for rnd,pair in enumerate(p['query_pairs']):
            if time.time()+p['minimum_cycle_seconds']>=learnstop:reason='NO_FULL_LEARNING_WINDOW';break
            before=L.state(q,opt);rollback=before;folder=OUT/'blocks'/str(rnd);folder.mkdir();queries=[];corrections=[]
            assert rnd==0 and pair==[0,1]
            for item in p['historical_control']['queries']:
                case,row=item['case'],item['row'];assert sha(row['path'])==row['sha256'] and row['success']
                assert row['policy_hash']==q.phash() and row['teacher'] and case['first']==p['teacher_first']
                a=N.S.arrays(row['path']);audit_query(q,a,case,p);queries.append(copy.deepcopy(item));corrections.append(query_group(q,a))
            record=dict(round=rnd,queries=queries,bc=None,rl=None,selected_success_ids=selected.copy());g['corrections']=corrections
            if not corrections:
                record['reason']='NO_VERIFIED_RECOVERY_LABELS';rounds.append(record);save(folder/'result.json',record);save(OUT/'rounds.json',rounds);rollback=None;reason=record['reason'];break
            shadow,so=learner(base,p);L.load(shadow,so,before);fit=fit_bc(q,opt,g,p,lg,event('BC'+str(rnd),'live'));checkfit=fit_bc(shadow,so,g,p,lg,event('BC'+str(rnd),'shadow'));assert fit==checkfit and L.statehash(q,opt)==L.statehash(shadow,so);model=export(q,opt,folder/'bc_proposed.pt.gz');check=joint(f'BC_{rnd}_proposal');accepted=keep(check) and check['left5']['success']>=1
            if not accepted:L.load(q,opt,before)
            else:selected=check['success_ids'];anyaccepted=True
            record['bc']=dict(fit=fit,proposed_model=model,checks=check,accepted=accepted,retained_hashes=L.statehash(q,opt))
            if accepted:
                opt=torch.optim.Adam(params(q),lr=p['lr']);rstate=L.state(q,opt);rollback=rstate;job=p['left5_jobs'][pair[0]];seeds=[[p['seed'],rnd,k] for k in range(4)];vals=evaluate([job]*4,f'train_{rnd}',full=True,seeds=seeds);evaluate([job]*4,f'train_{rnd}_exact',seeds=seeds,refs=[r for a,r in vals]);batch=[a for a,r in vals];shadow,so=learner(base,p);L.load(shadow,so,rstate);fit=fit_ppo(q,opt,batch,g,p,rnd,lg,event(rnd,'live'));checkfit=fit_ppo(shadow,so,batch,g,p,rnd,lg,event(rnd,'shadow'));assert fit==checkfit and L.statehash(q,opt)==L.statehash(shadow,so);model=export(q,opt,folder/'rl_proposed.pt.gz');check=joint(f'RL_{rnd}_proposal');rkeep=fit['numeric_accept'] and keep(check)
                if not rkeep:L.load(q,opt,rstate)
                else:selected=check['success_ids'];anyaccepted=True
                record['rl']=dict(fit=fit,proposed_model=model,checks=check,accepted=rkeep,retained_hashes=L.statehash(q,opt),sampling=tally([r for a,r in vals]),job=job,seeds=seeds)
            record['selected_success_ids']=selected.copy();rounds.append(record);save(folder/'result.json',record);save(OUT/'rounds.json',rounds);rollback=None
        result.update(model=export(q,opt,OUT/'candidate.pt.gz'),any_learning_accepted=anyaccepted,main54=None,old275=None,fresh16=None,reload78=None,stable_wide_six=False)
        centres=evaluate(p['centres'],'final_centres');result['main_centres']=tally([r for a,r in centres]);evaluate(p['centres'],'final_centres_exact',refs=[r for a,r in centres],model_path=result['model']['path']);result['final_joint']=joint('final_joint');assert set(selected)<=set(result['final_joint']['success_ids']);selected=result['final_joint']['success_ids']
        if anyaccepted and result['final_joint']['all14']:
            dev=[r for a,r in evaluate(p['main_jobs'],'main54')];old=[r for a,r in evaluate(base['final_old_jobs'],'old275')];result.update(main54=tally(dev),old275=tally(old))
            if all(r['success'] for r in [*dev,*old]):
                for f,h in p['fresh_ledger'].items():assert sha(f)==h
                fresh=[r for a,r in evaluate(p['fresh_jobs'],'fresh16')];result['fresh16']=tally(fresh)
                if all(r['success'] for r in fresh):
                    look={r['job_id']:r for r in [*dev,*old,*fresh]};reload=evaluate(p['reload_jobs'],'reload78',refs=[look[j['id']] for j in p['reload_jobs']],model_path=result['model']['path']);result.update(reload78=dict(exact=len(reload),independent_sample=False),stable_wide_six=len(reload)==78)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();exact=None
        if rollback is not None:L.load(q,opt,rollback);exact=digest(L.state(q,opt))==digest(rollback);export(q,opt,OUT/'interrupted_retained.pt.gz')
        save(OUT/'failure.json',dict(error=error,phase=phase,physics=sum(counts),commands=commands,updates=updates,interrupted_Actor_Adam_rollback=exact))
    finally:
        if pool:pool.close()
        signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();s=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),commands=commands,updates=updates,rounds=rounds,result=result,reason=reason,all_workers_stopped=True,all_native_sessions_closed=True,seven_eight_entered=False,narrow_paused=True);save(OUT/'summary.json',s);save(OUT/'status.json',dict(state=state,updated_epoch=time.time(),physics=sum(counts),commands=commands,updates=updates));N.manifest(OUT)
    assert state=='COMPLETE'
if __name__=='__main__':main()
