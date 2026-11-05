from night_common import *
import multiprocessing as mp

def main(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');configure(out);base=science(p);base.update(out=str(out))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert not (out/'summary.json').exists()
    assert read(Path(p['teacher'])/'audit.json')['teacher_safe_GOAL'] and read(Path(p['teacher'])/'summary.json')['teacher_safe_GOAL']
    for f,h in p['teacher_input_hashes'].items():assert sha(f)==h,f
    start=time.time();deadline=min(start+p['main_seconds'],p['native_deadline']);learnend=min(deadline-1800,p['learning_deadline'])
    state='FAILED';phase='INITIAL_DATA';last=0.;commands=0;calls={'live':0,'shadow':0};rows=[];replays=[];processes=[];pipes=[];results={};model=None;center_ref=None
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2)
    def status():
        nonlocal last
        assert time.time()<deadline-10
        if time.time()-last>3:
            last=time.time();capacity(out,p['final_bytes']);save(out/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,commands=commands,physics=sum(counts[:]),updates=calls,deadline=deadline))
    def guard():assert time.time()<learnend;status()
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def export(q,opt,path):
        v=dict(state=learned_state(q,opt),hashes=hashes_state(q,opt),policy_hash=q.phash(),supervised=True,RL_updates=0)
        b=io.BytesIO();torch.save(v,b);z=gzip.compress(b.getvalue(),mtime=0);assert len(z)<=384*1024;path.write_bytes(z)
        return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),hashes=v['hashes'])
    def event(replica,where,k):
        if where=='AFTER':calls[replica]+=1
        events.write((json.dumps([replica,where,k])+'\n').encode())
    def jobs(js,role,q,reference=None,readback=False):
        nonlocal commands,phase,center_ref
        phase=role;items=[]
        for offset in range(0,len(js),2):
            group=js[offset:offset+2];assert time.time()+40<deadline
            assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
            commands+=len(group);status()
            for pipe,j in zip(pipes,group):
                pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,learned=q.learned(),policy_hash=q.phash(),deadline_epoch=deadline-15,
                    readback_model=model['path'] if readback else None,stochastic=False))
            batch=[receive(pipe) for pipe in pipes[:len(group)]]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:
                    write_npz(out/'partial.npz',S.packed(a),512*1024);save(out/'partial_case.json',r);raise RuntimeError('Censored native evidence preserved')
                fullhash=hashes(a)
                if reference is not None:
                    prior=reference[r['job_id']];assert fullhash==prior['full_hashes'] and r['integration_sha256']==prior['integration_sha256']
                    rec=dict(job_id=r['job_id'],role=role,physics_steps=r['physics_steps'],full_hashes=fullhash,independent_sample=False,exact=True)
                    replays.append(rec)
                    with (out/'replays.jsonl').open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
                else:
                    data_={k:a[k] for k in ['actions','native_counts','end_flags']};path=out/'episodes'/f'{len(rows):04}.npz';write_npz(path,data_,8*1024)
                    rec={k:r[k] for k in ['job_id','role','success','collision','timeout','complete','physics_steps','remaining_m','integration_sha256']};rec.update(path=str(path),sha256=sha(path),full_hashes=fullhash)
                    rows.append(rec)
                    with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(rec,separators=(',',':'))+'\n')
                    if role=='final_new070' and r['job_id']==base['narrow_jobs'][0]['id']:
                        write_npz(out/'student_centre.npz',S.packed(a),512*1024);center_ref=rec
                items.append(rec)
            status()
        return items
    def count(items):return dict(success=sum(v['success'] for v in items),collision=sum(v['collision'] for v in items),timeout=sum(v['timeout'] for v in items),valid=len(items),invalid=0,censored=0)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('Student external stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Student deadline')));signal.alarm(max(1,int(deadline-start)-5));events=None
    try:
        q,opt=learner(base);groups=data(q,base,Path(p['teacher'])/'mixed.npz',guard);old=old_data(q,base,guard)
        initial=export(q,opt,out/'initial.pt.gz')
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=native_worker,args=(b,k,base,counts));proc.start();b.close();pipes.append(a);processes.append(proc)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(out/'workers.json',dict(parent=os.getpid(),workers=[x.pid for x in processes]))
        initialold=jobs(base['final_old_jobs'],'initial_old275',q);results['initial_old275']=count(initialold)
        initialdev=jobs([*base['old_jobs'],*base['narrow_jobs'],*base['original_jobs']],'initial_dev',q);results['initial_dev']=count(initialdev)
        phase='SUPERVISED_FULL_RECURRENT_BODY';status();before=learned_state(q,opt);shadow,shadowopt=learner(base);shadow.load_learned(before['learned']);shadowopt.load_state_dict(copy.deepcopy(before['optimizer']))
        with gzip.GzipFile(filename=str(out/'optimizer_calls.jsonl.gz'),mode='wb',mtime=0) as events:
            live=fit(q,opt,groups,1000,guard,lambda w,k:event('live',w,k),old)
            other=fit(shadow,shadowopt,groups,1000,guard,lambda w,k:event('shadow',w,k),old)
        assert equal(live,other) and hashes_state(q,opt)==hashes_state(shadow,shadowopt)
        write_npz(out/'trace.npz',dict(trace=live),96*1024);model=export(q,opt,out/'candidate.pt.gz')
        finalnew=jobs(base['narrow_jobs'],'final_new070',q);finalold20=jobs(base['old_jobs'],'final_old20',q);finalorig=jobs(base['original_jobs'],'final_original2',q)
        results.update(new070=count(finalnew),old20=count(finalold20),original2=count(finalorig),old275=None,originalsix54=None,fresh16=None,reload78=None,stable_original_six=False)
        center=finalnew[0];jobs([base['narrow_jobs'][0]],'centre_model_readback',q,{center['job_id']:center},True)
        if results['new070']['success']==5 and results['old20']['success']==20 and results['original2']['success']==2:
            oldall=jobs(base['final_old_jobs'],'final_old275',q);six=jobs(base['six_jobs'],'final_originalsix54',q);results.update(old275=count(oldall),originalsix54=count(six))
            if results['old275']['success']==275 and results['originalsix54']['success']==54:
                fresh=jobs(base['fresh_jobs'],'final_fresh16',q);results['fresh16']=count(fresh)
                if results['fresh16']['success']==16:
                    mapping={v['job_id']:v for v in [*oldall,*six,*fresh]};rr=jobs(base['reload_jobs'],'independent_native_reload78',q,mapping,True)
                    results.update(reload78=dict(exact=len(rr)==78),stable_original_six=len(rr)==78)
        state='COMPLETE'
    except BaseException:save(out/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics=sum(counts[:]),updates=calls))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,BrokenPipeError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
        summary=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,commands=commands,physics=sum(counts[:]),updates=calls,models=model,result=results,all_workers_stopped=all(not x.is_alive() for x in processes),teacher_source=p['teacher'],supervised=True,new_RL_updates=0,
            files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name not in ['main.log','status.json']})
        save(out/'summary.json',summary);save(out/'status.json',dict(state=state,updated_epoch=time.time(),commands=commands,physics=sum(counts[:]),updates=calls));capacity(out,p['final_bytes']);print(json.dumps(dict(state=state,result=results)))
    assert state=='COMPLETE'

def hashes_state(q,opt):return {k:digest(v) for k,v in learned_state(q,opt).items()}
def audit(outpath):
    offline();out=Path(outpath);p=read(out/'protocol.json');base=science(p);s=read(out/'summary.json');stopped(out);assert s['state']=='COMPLETE' and not (out/'audit.json').exists()
    for pid in read(out/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    for f,h in p['teacher_input_hashes'].items():assert sha(f)==h,f
    torch.set_num_threads(1);q,opt=learner(base);groups=data(q,base,Path(p['teacher'])/'mixed.npz',lambda:None);old=old_data(q,base,lambda:None)
    trace=fit(q,opt,groups,1000,lambda:None,lambda *_:None,old);assert equal(trace,load_npz(out/'trace.npz')['trace'])
    saved=torch.load(io.BytesIO(gzip.decompress((out/'candidate.pt.gz').read_bytes())),weights_only=False,map_location='cpu');assert hashes_state(q,opt)==saved['hashes']==s['models']['hashes']
    events=[json.loads(x) for x in gzip.decompress((out/'optimizer_calls.jsonl.gz').read_bytes()).splitlines()];assert len(events)==4000 and s['updates']==dict(live=1000,shadow=1000)
    rows=[json.loads(x) for x in (out/'cases.jsonl').read_text().splitlines()];reps=[json.loads(x) for x in (out/'replays.jsonl').read_text().splitlines()]
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=load_npz(row['path']);assert int(a['native_counts'].sum())==row['physics_steps'] and bool(a['end_flags'][-1,1])==row['success']
    assert s['commands']==len(rows)+len(reps) and s['physics']==sum(v['physics_steps'] for v in [*rows,*reps])
    for path,h in s['files'].items():assert sha(out/path)==h
    save(out/'audit.json',dict(state='PASS',native=0,new_Actor_updates=0,discarded_historical_Adam=1000,exact_live_shadow_and_independent_Actor_Adam_trace=True,all_cases_replay_raw_counts_and_SHA=True,physics=s['physics'],commands=s['commands'],stable_original_six=bool(s['result'].get('stable_original_six')),teacher_source=p['teacher']));print('STUDENT_INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
