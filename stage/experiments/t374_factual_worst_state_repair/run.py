from t374_core import *
import ast


def prepare():
    assert not OUT.exists() and read(DOC/'T373_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE_AND_FULL_REMOTE_BACKUP'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T373.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t372_nonlinear_terminal_repair/protocol.json'))
    p.update(id='T374',folder=OUT.name,out=str(OUT),experiment='T374_FACTUAL_PARENT_WORST_STATE_REPAIR',arms=['factual_worst_state'],
        factual_parent_rows=read(ROOT/'stage/runs/t373_factual_parent_support/cases.json'),wall_seconds=1200,final_reserve_seconds=300,audit_seconds=300,all_work_seconds=2400,
        max_commands=2,max_physics=80000,geometry_commands=0,final_bytes=3*1024**2,full_bytes=384*1024,partial_bytes=384*1024,model_bytes=256*1024,
        max_live_Adam=300,max_shadow_Adam=300,preflight_discarded_Adam=8,audit_historical_discarded_Adam=300,fit_steps=300,
        contract=(DOC/'T374_PLAN.md').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    initial=NightLearner(p);data=dataset(p,initial.actor,lambda:None);reference=NightActor(p).eval().requires_grad_(False)
    refmodel=read(ROOT/'stage/runs/t372_nonlinear_terminal_repair/summary.json')['model'];reference.load_learned(read_model(refmodel['path']))
    p['reference_model']=refmodel;p['reference_metrics']=locality(reference,data)
    p['diagnostic_gate_limits']={name:p['reference_metrics'][name]/(1. if name=='patch_steer_rmse' else 4.) for name in ['patch_steer_rmse','prefix248_executed_steer_max','first20_executed_steer_max','factual_executed_steer_max']}
    assert all(v>0. for v in p['diagnostic_gate_limits'].values()) and len(p['factual_parent_rows'])==14
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T374_PLAN.md'),str(DOC/'T373_FINDINGS.md'),str(DOC/'T373_CLOSEOUT.json'),str(DOC/'NEXT_AFTER_T373.md'),str(ROOT/'stage/runs/t373_factual_parent_support/cases.json'),str(ROOT/'stage/runs/t373_factual_parent_support/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T373.json'),refmodel['path'],str(ROOT/'stage/runs/t372_nonlinear_terminal_repair/summary.json')}
    files|={r['path'] for r in p['factual_parent_rows']};p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    used=K.S.used(ROOT/'stage/runs');worst=768*1024+2*p['model_bytes']+p['full_bytes']+p['partial_bytes']+128*1024+128*1024
    assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T374_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=2,max_physics=80000,live_Adam=300,shadow_Adam=300,historical_audit_Adam=300,conditional_native_only_after_gate=True))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,diagnostic_gate_limits=p['diagnostic_gate_limits'],full_global_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);report=contract(p,obj,data);initial=obj.state();calls=0;fixed=fixed_hash(obj.actor)
    for rnd in range(2):
        before=obj.state();immutable=digest(before);shadow=NightLearner(p);shadow.load(before)
        def event(replica,r):
            with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,r['phase'],r['step']])+'\n')
        a=fit(p,obj,data,lambda:None,lambda r:event('live',r),steps=2);calls+=2
        b=fit(p,shadow,data,lambda:None,lambda r:event('shadow',r),steps=2);calls+=2
        assert a==b and obj.hashes()==shadow.hashes() and digest(before)==immutable and fixed_hash(obj.actor)==fixed
        if rnd==1:obj.load(before);assert digest(obj.state())==immutable
    obj.load(initial);assert digest(obj.state())==digest(initial) and calls==8
    reference=NightActor(p).eval().requires_grad_(False);reference.load_learned(read_model(p['reference_model']['path']))
    assert locality(reference,data)==p['reference_metrics'] and not gate(p,p['reference_metrics'])['passed']
    assert not gate(p,locality(obj.actor,data))['passed']
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=0,discarded_Adam=calls,contract=report,mature_shadow_and_rollback_exact=True,frozen_parent_immutable=True,reference_and_zero_initial_fail_joint_gate=True))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=0,discarded_Adam=8)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];learn_end=deadline-300
    ctx=mp.get_context('spawn');counts=ctx.Array('q',1);processes=[];pipes=[];rows=[];replays=[];commands=0;phase='DATASET';last=0.;state='FAILED';metrics={};updates=dict(live=0,shadow=0);model=None
    obj=NightLearner(p);initial=obj.state();meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(counts[:]),updates=updates))
    def guard():assert time.time()<learn_end;status();meter.check()
    def event(replica,r):
        if r['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([replica,r['phase'],r['step']])+'\n')
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    def centre(role):
        nonlocal commands,phase
        phase=role;commands+=1;assert commands<=2 and sum(counts[:])+40000<=p['max_physics'];status(True)
        j=p['narrow_jobs'][0];pipes[0].send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm='memory',rho=0.,learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=model['path'] if role=='model_readback' else None,stochastic=False))
        item=receive(pipes[0]);assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays']
        if not r['complete']:
            buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Censored raw preserved; no retry')
        if role=='centre':
            buf=io.BytesIO();np.savez_compressed(buf,**packed(a));blob=buf.getvalue();assert len(blob)<=p['full_bytes'];path=OUT/'centre.npz';path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path));rows.append(r);save(OUT/'cases.json',rows)
        else:
            ref=arrays(rows[0]['path'])
            assert r['array_hashes']==rows[0]['array_hashes'] and r['integration_sha256']==rows[0]['integration_sha256'] and a.keys()==ref.keys() and all(np.array_equal(v,ref[k]) for k,v in a.items())
            r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);replays.append(r);save(OUT/'replays.json',replays)
        return r
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T374 deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        data=dataset(p,obj.actor,guard);fallback=export(p,obj,OUT/'initial_fallback.pt.gz');initial_contract=contract(p,obj,data)
        save(OUT/'data_contract.json',dict(contract=initial_contract,fallback=fallback,source_routes=14,source_decisions=2639,schedule_sha256=digest(schedule(p,data[0])),new_RL_updates=0))
        before=obj.state();immutable=digest(before);input_hashes=obj.hashes();shadow=NightLearner(p);shadow.load(before);phase='FIT';status(True)
        try:
            trace=fit(p,obj,data,guard,lambda r:event('live',r));other=fit(p,shadow,data,guard,lambda r:event('shadow',r));assert trace==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
        except BaseException:
            obj.load(before);save(OUT/'interrupted.json',dict(input_hashes=input_hashes,restored_hashes=obj.hashes(),retained_updates=0));raise
        save(OUT/'trace.json',trace);save(OUT/'block.json',dict(input_hashes=input_hashes,output_hashes=obj.hashes(),live_Adam=300,shadow_Adam=300,trace_sha256=digest(trace),exact_shadow=True))
        metrics=locality(obj.actor,data);decision=gate(p,metrics);save(OUT/'metrics.json',metrics);save(OUT/'gate.json',decision);model=export(p,obj,OUT/'candidate.pt.gz')
        if decision['passed']:
            assert time.time()+180<deadline;phase='NATIVE_START';status(True);a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1';proc=ctx.Process(target=night_worker,args=(b,0,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a);os.environ['STAGE_PROCESS_CPU']='0'
            ready=receive(a);assert ready['ready'] and ready['affinity']==[1];save(OUT/'workers.json',dict(parent=os.getpid(),workers=[proc.pid],ready=[ready]))
            centre('centre');centre('model_readback')
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
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,model=model,metrics=metrics,
        diagnostic_gate=read(OUT/'gate.json') if (OUT/'gate.json').exists() else None,centre_result=rows[0] if rows else None,native_model_readbacks=len(replays),all_workers_stopped=all(not v.is_alive() for v in processes),research_candidate_not_promoted=True,new_RL_updates=0,new_native_training_collection=0,old275=None,six54=None,fresh16=None,reload78=None,stable_original_six=False,entered7or8=False)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T374 audit300s')));signal.alarm(300);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);initial_contract=contract(p,obj,data);assert initial_contract==read(OUT/'data_contract.json')['contract']
    fallback=read(OUT/'data_contract.json')['fallback'];assert K.C.sha(fallback['path'])==fallback['sha256']
    fallback_state=torch.load(io.BytesIO(gzip.decompress(Path(fallback['path']).read_bytes())),weights_only=False,map_location='cpu')['state'];assert digest(fallback_state)==digest(obj.state())
    block=read(OUT/'block.json');assert obj.hashes()==block['input_hashes'];events=[];trace=fit(p,obj,data,lambda:None,lambda r:events.append(r))
    assert trace==read(OUT/'trace.json') and digest(trace)==block['trace_sha256'] and obj.hashes()==block['output_hashes'] and fixed_hash(obj.actor)==p['repair_parent']['hashes']['learned']
    recorded=[json.loads(v) for v in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for replica in ['live','shadow']:assert [dict(phase=v[1],step=v[2]) for v in recorded if v[0]==replica]==events
    assert s['updates']==dict(live=300,shadow=300);assert locality(obj.actor,data)==read(OUT/'metrics.json')==s['metrics'];decision=gate(p,s['metrics']);assert decision==read(OUT/'gate.json')==s['diagnostic_gate']
    model=s['model'];assert K.C.sha(model['path'])==model['sha256'];z=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu');assert digest(z['state'])==digest(obj.state())
    physics=0
    if decision['passed']:
        rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==1
        r=rows[0];a=arrays(r['path']);assert K.C.sha(r['path'])==r['sha256'] and {k:digest(v) for k,v in a.items()}==r['array_hashes'];q=obj.actor;q.reset_noise();q.sample=False;h=None
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        with torch.no_grad():
            for k,x in enumerate(a['states']):
                assert np.array_equal(np.zeros(16) if h is None else h[0,0].numpy(),a['hidden_before'][k]);action,_,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(action[0].numpy(),a['actions'][k])
                for name,value in q.tape[-1].items():assert np.array_equal(value,a[name][k])
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);disc=K.R.GAMMA**n;d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
                reward=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not(c or g or t))
                if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
                assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256'] and int(a['native_counts'].astype(int).sum())==r['physics_steps']
        rep=reps[0];assert rep['array_hashes']==r['array_hashes'] and rep['integration_sha256']==r['integration_sha256'] and rep['all_arrays_hidden_history_integration_exact'];physics=r['physics_steps']+rep['physics_steps'];assert s['commands']==2
    else:assert s['commands']==s['physics_steps']==s['native_model_readbacks']==0 and s['centre_result'] is None and read(OUT/'workers.json')['workers']==[]
    assert physics==s['physics_steps']<=p['max_physics'] and s['elapsed_s']<=1200 and not (OUT/'partial.npz').exists()
    result=dict(state='PASS',new_native=0,historical_discarded_Adam=300,source_model_arrays_full_history_exact=True,all_parent_bytes_immutable=True,live_shadow_final_ActorAdam_reconstructed=True,diagnostic_gate=decision,native_commands=s['commands'],physics_steps=physics,no_promotion_or_stable_six_claim=True)
    save(OUT/'audit.json',result);save(DOC/'T374_AUDIT.json',result);signal.alarm(0);print(json.dumps(result))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T374_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage learned terminal steering repair',
        'timeout','--signal=TERM','--kill-after=1s','2400s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+1200,all_work_deadline_epoch=epoch+2400,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T374_DRIVER.json').exists();active=None
    save(DOC/'T374_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='NATIVE'))
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
    def stop(*_):raise TimeoutError('Bounded T374 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'run.py'),'main'],min(1200,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T374_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'run.py'),'audit'],min(300,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T374_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(HERE/'transport.py'),'T374'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T374_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T374.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T374_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T374.json')))
    except BaseException:
        save(DOC/'T374_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()
