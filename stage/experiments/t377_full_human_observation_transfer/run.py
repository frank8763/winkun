from t377_core import *
import ast,importlib.util


def prepare():
    assert not OUT.exists() and read(ROOT/'stage/runs/t376_cold_trace_reserve/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T376.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    previous=ROOT/'stage/runs/t375_observation_residual/protocol.json';p=copy.deepcopy(read(previous))
    for key in ['diagnostic_gate_limits','reference_metrics','reference_model']:p.pop(key,None)
    jobs={j['id']:j for j in [*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs']]}
    development=[jobs[r['job_id']] for r in p['factual_parent_rows']]+p['narrow_jobs'][1:]
    assert len(development)==18 and len({j['id'] for j in development})==18
    p.update(id='T377',folder=OUT.name,out=str(OUT),experiment='T377_FULL_HUMAN_DRIVE_STEER_OBSERVATION_TRANSFER',development_jobs=development,
        wall_seconds=1500,final_reserve_seconds=900,audit_seconds=300,all_work_seconds=2700,max_commands=38,max_physics=1520000,
        final_bytes=int(2.5*1024**2),model_bytes=128*1024,full_bytes=384*1024,partial_bytes=384*1024,compact_bytes=8*1024,
        contract=(DOC/'T377_PLAN.md').read_text(),arms=['full_human_both_commands'],human_valid_labels=572)
    torch.set_num_threads(1);parent=BASE.NightActor(p,0.).eval().requires_grad_(False);parent.load_learned(read_model(p['repair_parent']['path']));features=[]
    with torch.no_grad():
        for path,first in [(p['demonstration'],0),(p['correction'],20)]:
            x=torch.tensor(arrays(path)['states']);s=((x.double()-parent.mean)/parent.scale).clamp(-10,10);h=parent.core(parent.encoder(s)[:,None])[0][:,0];features.append(torch.cat([h,s],1)[first:])
        mean=torch.cat(features).mean(0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['adapter_seed']);layer=torch.nn.Linear(136,4).double();p['adapter_initial_bias']=(-layer.weight@mean).tolist()
    p['initialization']='Fixed seed; bias centred on572valid human features; last4x2layer zero, no fit/search'
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T377_PLAN.md'),str(DOC/'NEXT_AFTER_T375.md'),str(ROOT/'stage/runs/t376_cold_trace_reserve/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T376.json'),str(ROOT/'stage/experiments/t373_factual_parent_support/run.py')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');worst=768*1024+2*p['model_bytes']+p['full_bytes']+p['partial_bytes']+18*p['compact_bytes']+128*1024+128*1024+64*1024
    assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'T377_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=38,max_physics=1520000,live_Adam=300,shadow_Adam=300,historical_audit_Adam=300))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,full_global_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);report=contract(p,obj,data);initial=obj.state();calls=0
    for rnd in range(2):
        before=obj.state();immutable=digest(before);shadow=NightLearner(p);shadow.load(before)
        def event(replica,r):
            with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,r['phase'],r['step']])+'\n')
        a=fit(p,obj,data,lambda:None,lambda r:event('live',r),steps=2);calls+=2;b=fit(p,shadow,data,lambda:None,lambda r:event('shadow',r),steps=2);calls+=2
        assert a==b and obj.hashes()==shadow.hashes() and digest(before)==immutable
        assert torch.count_nonzero(obj.actor.adapter[0].weight.grad[:,16:])>0
        if rnd==1:obj.load(before);assert digest(obj.state())==immutable
    obj.load(initial);assert digest(obj.state())==digest(initial) and calls==8
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry=0,discarded_Adam=8,contract=report,current_observation_weight_gradients_nonzero=True,mature_shadow_rollback_exact=True));print('PREFLIGHT_PASS')


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];learn_end=deadline-900;ctx=mp.get_context('spawn');counts=ctx.Array('q',2);processes=[];pipes=[];rows=[];replays=[];commands=0;phase='DATASET';last=0.;state='FAILED';model=None;updates=dict(live=0,shadow=0);result={}
    obj=NightLearner(p);initial=obj.state();meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes']);save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5;save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,commands=commands,physics_steps=sum(counts[:]),updates=updates))
    def guard():assert time.time()<learn_end;status();meter.check()
    def event(replica,r):
        if r['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([replica,r['phase'],r['step']])+'\n')
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def preserve_partial(r,a,why):
        buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError(why)
    def jobs(js,role,reference=None):
        nonlocal commands,phase
        phase=role;items=[]
        for offset in range(0,len(js),2):
            group=js[offset:offset+2];assert commands+len(group)<=38 and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics'] and time.time()+40<deadline;commands+=len(group);status(True)
            for pipe,j in zip(pipes,group):pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=model['path'] if reference is not None else None,stochastic=False))
            batch=[receive(pipe) for pipe in pipes[:len(group)]]
            for item in batch:
                assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
                if not r['complete']:preserve_partial(r,a,'Censored native raw preserved; no retry')
                if reference is None:
                    full=r['job_id']=='narrow070_dev_0';payload=packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']};buf=io.BytesIO();np.savez_compressed(buf,**payload);blob=buf.getvalue()
                    if len(blob)>p['full_bytes' if full else 'compact_bytes']:preserve_partial(r,a,'Native output exceeds fixed cap')
                    path=OUT/(r['job_id']+'.npz');assert not path.exists();path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full,saved_array_hashes={k:digest(v) for k,v in payload.items()});rows.append(r);save(OUT/'cases.json',rows)
                else:
                    ref=reference[r['job_id']]
                    if not(r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256'] and a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())):preserve_partial(r,a,'Saved-model readback mismatch preserved')
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r);save(OUT/'replays.json',replays)
                items.append(item)
            status(True)
        return items
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T377 main deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        data=dataset(p,obj.actor,guard);initial_contract=contract(p,obj,data);fallback=export(p,obj,OUT/'initial_fallback.pt.gz');result['initial_fit_metrics']=metrics(obj.actor,data);save(OUT/'data_contract.json',dict(contract=initial_contract,fallback=fallback,schedule_sha256=digest(schedule(p,data[0]))))
        before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=NightLearner(p);shadow.load(before);phase='FIT';status(True)
        try:
            trace=fit(p,obj,data,guard,lambda r:event('live',r));other=fit(p,shadow,data,guard,lambda r:event('shadow',r));assert trace==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
        except BaseException:obj.load(before);save(OUT/'interrupted.json',dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
        save(OUT/'trace.json',trace);save(OUT/'block.json',dict(input_hashes=input_hashes,output_hashes=obj.hashes(),trace_sha256=digest(trace),live_Adam=300,shadow_Adam=300));result['final_fit_metrics']=metrics(obj.actor,data);model=export(p,obj,OUT/'candidate.pt.gz')
        assert time.time()+180<deadline;phase='NATIVE_START';status(True)
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=night_worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready));save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        centre=jobs([p['narrow_jobs'][0]],'centre');jobs([p['narrow_jobs'][0]],'centre_readback',{v['row']['job_id']:v for v in centre});result['centre_safe_GOAL']=centre[0]['row']['success']
        if result['centre_safe_GOAL']:
            dev=jobs(p['development_jobs'],'development');jobs(p['development_jobs'],'development_readback',{v['row']['job_id']:v for v in dev})
            old_ids={r['job_id'] for r in p['factual_parent_rows']};reg=[v['row'] for v in dev if v['row']['job_id'] in old_ids];narrow=[centre[0]['row'],*[v['row'] for v in dev if v['row']['job_id'].startswith('narrow070')]]
            result['parent14']={k:sum(bool(r[k]) for r in reg) for k in ['success','collision','timeout']};result['new0705']={k:sum(bool(r[k]) for r in narrow) for k in ['success','collision','timeout']};result['pilot_retention_and_development_pass']=all(r['success'] for r in [*reg,*narrow])
        else:result.update(parent14=None,new0705=None,pilot_retention_and_development_pass=False)
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    if model is None:obj.load(initial);model=export(p,obj,OUT/'candidate.pt.gz')
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,model=model,result=result,primary_cases=len(rows),native_model_readbacks=len(replays),all_workers_stopped=all(not v.is_alive() for v in processes),new_RL_updates=0,new_native_training_collection=0,research_candidate_not_promoted=True,old275=None,six54=None,fresh16=None,reload78=None,stable_original_six=False,entered7or8=False)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T377 audit300s')));signal.alarm(300);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);assert contract(p,obj,data)==read(OUT/'data_contract.json')['contract'];assert metrics(obj.actor,data)==s['result']['initial_fit_metrics']
    fallback=read(OUT/'data_contract.json')['fallback'];assert K.C.sha(fallback['path'])==fallback['sha256'];state=torch.load(io.BytesIO(gzip.decompress(Path(fallback['path']).read_bytes())),weights_only=False,map_location='cpu')['state'];assert digest(state)==digest(obj.state())
    block=read(OUT/'block.json');assert obj.hashes()==block['input_hashes'];ev=[];trace=fit(p,obj,data,lambda:None,lambda r:ev.append(r));assert trace==read(OUT/'trace.json') and digest(trace)==block['trace_sha256'] and obj.hashes()==block['output_hashes'] and fixed_hash(obj.actor)==p['repair_parent']['hashes']['learned']
    recorded=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:assert [dict(phase=v[1],step=v[2]) for v in recorded if v[0]==replica]==ev
    assert metrics(obj.actor,data)==s['result']['final_fit_metrics'];model=s['model'];assert K.C.sha(model['path'])==model['sha256'];state=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')['state'];assert digest(state)==digest(obj.state())
    rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');expected=19 if s['result']['centre_safe_GOAL'] else 1;assert len(rows)==len(reps)==expected and s['commands']==2*expected;physics=0
    spec=importlib.util.spec_from_file_location('stage_t377_native_forward_checker',ROOT/'stage/experiments/t373_factual_parent_support/run.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert int(a['native_counts'].astype(int).sum())==r['physics_steps'];physics+=r['physics_steps']
        with np.load(r['path']) as z:assert {k:digest(z[k]) for k in z.files}==r['saved_array_hashes']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any() and not r['administrative_censor'];c,g,t=map(bool,a['end_flags'][-1]);assert r['collision']==c and r['success']==(g and not c) and r['timeout']==t
        if r['full_arrays_saved']:checker.check_forward(obj.actor,a,r)
    for r,ref in zip(reps,rows):assert r['job_id']==ref['job_id'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'] and r['all_arrays_hidden_history_integration_exact'];physics+=r['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics'] and s['elapsed_s']<=1500 and s['updates']==dict(live=300,shadow=300) and not (OUT/'partial.npz').exists()
    result=dict(state='PASS',new_native=0,historical_discarded_Adam=300,whole_human_mask_and_current_history_exact=True,all_parent_bytes_immutable=True,live_shadow_final_ActorAdam_reconstructed=True,raw_reward_history_arrays_and_model_readbacks_verified=True,commands=s['commands'],physics_steps=physics,no_promotion_or_original_six_stability_claim=True)
    save(OUT/'audit.json',result);save(DOC/'T377_AUDIT.json',result);signal.alarm(0);print(json.dumps(result))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T377_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage learned terminal steering repair',
        'timeout','--signal=TERM','--kill-after=1s','2700s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+1500,all_work_deadline_epoch=epoch+2700,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T377_DRIVER.json').exists();active=None
    save(DOC/'T377_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='MAIN_FULL_HUMAN_TRANSFER_THEN_NATIVE'))
    def call(args,seconds,log):
        nonlocal active
        with log.open('xb') as f:
            active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),*args],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=active.wait(timeout=seconds)
                if rc:raise subprocess.CalledProcessError(rc,str(args))
            except BaseException:
                # Reap the complete subprocess group even if its leader exited.
                try:os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                if active.poll() is None:
                    try:active.wait(1)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.wait();raise
            finally:active=None
    def stop(*_):raise TimeoutError('Bounded T377 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'run.py'),'main'],min(1500,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T377_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'run.py'),'audit'],min(300,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T377_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(HERE/'transport.py'),'T377'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T377_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T377.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T377_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T377.json')))
    except BaseException:
        save(DOC/'T377_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()
