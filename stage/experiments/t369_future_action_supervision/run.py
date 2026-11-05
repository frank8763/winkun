"""One preregistered matched comparison, bounded native checks and unique closure."""
from t369_core import *
import ast,contextlib
import mujoco as mj


def prepare():
    assert not OUT.exists() and read(DOC/'T368_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE'
    assert read(ROOT/'stage/runs/t368_driving_steer_correlation/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T368.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t365_correction_aggregation/protocol.json'));prior=read(ROOT/'stage/runs/t365_correction_aggregation/summary.json')
    rows=read(ROOT/'stage/runs/t365_correction_aggregation/cases.json');factual=read(ROOT/'stage/runs/t366_factual_history/cases.json')
    p.update(id='T369',folder=OUT.name,out=str(OUT),experiment='T369_MATCHED_FUTURE_ACTION_SUPERVISION',initial_model=prior['model']['path'],initial_is_T349=False,
        initial_expected_hashes={r['job_id']:r['array_hashes'] for r in rows if r['role']=='final'},factual_rehearsal=factual,
        arms={'current':0.,'future':.5},horizon=5,fit_steps=600,fit_seed=20261105362,actor_lr=.001,geometry_commands=3,
        wall_seconds=3600,final_reserve_seconds=1200,audit_seconds=900,all_work_seconds=5400,max_commands=163,max_physics=6520000,
        final_bytes=12*1024**2,full_bytes=768*1024,compact_bytes=8*1024,partial_bytes=768*1024,model_bytes=512*1024,
        max_full_cases=3,max_compact_cases=93,max_live_Adam=1200,max_shadow_Adam=1200,preflight_discarded_Adam=16,audit_historical_discarded_Adam=1200,
        contract=(DOC/'T369_PLAN.md').read_text(),new_RL_updates=0,new_native_training_collection=0)
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{p['initial_model'],str(DOC/'T369_PLAN.md'),str(DOC/'T368_CLOSEOUT.json'),str(DOC/'T368_FINDINGS.md'),str(ROOT/'stage/runs/t368_driving_steer_correlation/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T368.json'),str(ROOT/'stage/experiments/t260_distance_archive/backup.py')}
    files|={r['path'] for r in factual};p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    used=K.S.used(ROOT/'stage/runs');worst=3*p['full_bytes']+93*p['compact_bytes']+3*p['model_bytes']+3*p['partial_bytes']+2*1024**2+256*1024
    assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T369_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=163,max_physics=6520000,live_supervised_Adam=1200,shadow=1200,preflight_discarded=16,audit_historical=1200))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,full_global_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p);data=dataset(p,obj.actor,lambda:None);report=contracts(p,obj,data);initial=obj.state();calls=0
    for arm in p['arms']:
        obj=NightLearner(p);assert digest(obj.state())==digest(initial)
        for rnd in range(2):
            before=obj.state();immutable=digest(before);shadow=NightLearner(p);shadow.load(before)
            def event(replica,r):
                with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([arm,rnd,replica,r['phase'],r['step']])+'\n')
            a=fit(p,obj,data,arm,lambda:None,lambda r:event('live',r),steps=2);calls+=2
            b=fit(p,shadow,data,arm,lambda:None,lambda r:event('shadow',r),steps=2);calls+=2
            assert a==b and obj.hashes()==shadow.hashes() and digest(before)==immutable
            if rnd==1:obj.load(before);assert digest(obj.state())==immutable
        obj.load(initial);assert digest(obj.state())==digest(initial)
    assert calls==16;folder=OUT/'preflight';folder.mkdir();geos=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE)
        try:
            for j,gap in zip([p['narrow_jobs'][0],p['course_dev_jobs'][0],p['original_jobs'][0]],[.7,1.17,.57]):
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j.get('circle_shift_x_m',.6);env,obs,_=K.R.task_env(create,j)
                try:
                    obs=K.C.perturb_anchor(env,obs,j);g=P.C.B.geometry(env,obs,w,j['case']['step_cap']);c=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
                    actual=float(env.model.geom_pos[c,0]-env.model.geom_size[c,0]-env.model.geom_pos[wall,0]-env.model.geom_size[wall,0]);assert abs(actual-gap)<1e-6 and g['valid'] and env.data.time==0. and env.step_count==0 and env.max_wheel==5.
                    assert np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]));geos.append(dict(id=j['id'],gap=actual,geometry=g))
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=3,discarded_Adam=calls,contracts=report,mature_shadow_and_rollback_exact=True,geometries=geos,budget=read(DOC/'T369_BUDGET.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=3,discarded_Adam=16)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];learn_end=deadline-1200
    ctx=mp.get_context('spawn');counts=ctx.Array('q',2);procs=[];pipes=[];rows=[];reps=[];commands=3;phase='STARTING';last=0.;active=[];updates=dict(live=0,shadow=0);models={};metrics={};blocks={};state='FAILED'
    objs={arm:NightLearner(p) for arm in p['arms']};initial=objs['current'].state();assert all(digest(o.state())==digest(initial) for o in objs.values())
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,updated_epoch=last,deadline_epoch=deadline,learning_deadline_epoch=learn_end,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    def guard():assert time.time()<learn_end;status();meter.check()
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    js=[*p['old_jobs'],*p['course_dev_jobs'],*p['original_jobs'],*p['narrow_jobs']]
    def jobs(role,obj,arm,reference=None,readback=None):
        nonlocal phase,commands,active
        phase=role;items=[]
        for off in range(0,32,2):
            group=js[off:off+2];assert commands+len(group)<=163 and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics'] and time.time()+60<deadline
            commands+=len(group);active=[j['id'] for j in group];status(True)
            for pipe,j in zip(pipes,group):pipe.send(dict(j,id=role+'_'+j['id'],source_reset_id=j['id'],role=role,arm=arm,learned=obj.actor.learned(),policy_hash=obj.actor.phash(),deadline_epoch=deadline-15,readback_model=readback))
            batch=[receive(pipe) for pipe in pipes];active=[]
            for it in batch:
                assert 'error' not in it,it.get('error');r,a=it['row'],it['arrays']
                if not r['complete']:
                    b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob);save(OUT/'partial_case.json',r);raise RuntimeError('Censored partial preserved; no retry')
                if role=='initial':
                    expected=p['initial_expected_hashes'][r['job_id']];assert {k:r['array_hashes'][k] for k in expected}==expected
                if reference is not None:
                    ref=reference[r['job_id']];assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                    assert a.keys()==ref['arrays'].keys() and all(np.array_equal(a[k],ref['arrays'][k]) for k in a)
                    r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);reps.append(r);save(OUT/'replays.json',reps)
                else:
                    full=r['job_id']=='narrow070_dev_0';data=packed(a) if full else {k:a[k] for k in ['actions','native_counts','end_flags']};b=io.BytesIO();np.savez_compressed(b,**data);blob=b.getvalue();assert len(blob)<=p['full_bytes' if full else 'compact_bytes']
                    path=OUT/'episodes'/(r['id']+'.npz');assert not path.exists();path.write_bytes(blob);r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=full,saved_array_hashes={k:digest(v) for k,v in data.items()});rows.append(r);save(OUT/'cases.json',rows)
                items.append(it)
            status(True)
        return items
    def score(items):
        return {name:{k:sum(bool(it['row'][k]) for it in items[sl]) for k in ['success','collision','timeout']} for name,sl in [('old20',slice(0,20)),('wide5',slice(20,25)),('original2',slice(25,27)),('new0705',slice(27,32))]}
    def success_sets(items):return {name:sorted(it['row']['job_id'] for it in items[sl] if it['row']['success']) for name,sl in [('old',slice(0,20)),('wide',slice(20,25)),('narrow',slice(27,32))]}
    def event(arm,replica,r):
        if r['phase']=='AFTER':updates[replica]+=1
        with (OUT/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([arm,replica,r['phase'],r['step']])+'\n')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T369 native deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);pr=ctx.Process(target=night_worker,args=(b,k,p,counts));pr.start();b.close();procs.append(pr);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready));save(OUT/'workers.json',dict(parent=os.getpid(),workers=[pr.pid for pr in procs],ready=ready))
        checked=jobs('initial',objs['current'],'common');base_sets=success_sets(checked);metrics['initial']=score(checked);models['fallback']=export(p,objs['current'],OUT/'initial_fallback.pt.gz')
        phase='DATASET';status(True);data=dataset(p,objs['current'].actor,guard);save(OUT/'data_contract.json',dict(human_valid_by_horizon=[targets(g)[1].sum(0).tolist() for g in data[1]],old211=len(data[0]),wide5=len(data[2]),narrow2=len(data[3]),schedule_sha256=digest(schedule(p,data[0])),same_initial_both_arms=True,current_common_initial_checks_are_reused=True,new_RL_updates=0))
        for arm,obj in objs.items():
            phase='FIT_'+arm;status(True);before=obj.state();immutable=digest(before);shadow=NightLearner(p);shadow.load(before)
            try:
                trace=fit(p,obj,data,arm,guard,lambda r:event(arm,'live',r));other=fit(p,shadow,data,arm,guard,lambda r:event(arm,'shadow',r))
                assert trace==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
            except BaseException:
                obj.load(before);save(OUT/('interrupted_'+arm+'.json'),dict(restored_hashes=obj.hashes(),retained_updates=0));raise
            save(OUT/('trace_'+arm+'.json'),trace);blocks[arm]=dict(input_hashes={k:digest(v) for k,v in before.items()},output_hashes=obj.hashes(),trace_sha256=digest(trace),live_Adam=600,shadow_Adam=600,exact_shadow=True);save(OUT/'blocks.json',blocks)
            models[arm]=export(p,obj,OUT/(arm+'_candidate.pt.gz'));metrics[arm]=dict(fit_error=descriptive_error(obj.actor,data))
        for arm,obj in objs.items():
            final=jobs('final_'+arm,obj,arm);sets=success_sets(final);retain=all(set(base_sets[k])<=set(sets[k]) for k in base_sets)
            metrics[arm].update(final=score(final),success_ids=sets,retention_pass=retain,research_only_rejected=not retain,source_fit_preserved=True)
            jobs('model_readback_'+arm,obj,arm,{it['row']['job_id']:it for it in final},models[arm]['path'])
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),phase=phase,commands=commands,physics_steps=sum(counts[:]),updates=updates,active=active))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for pr in procs:
            pr.join(5)
            if pr.is_alive():pr.kill();pr.join(3)
        signal.alarm(0)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),updates=updates,models=models,result=metrics,primary_cases=len(rows),native_model_readbacks=len(reps),all_workers_stopped=all(not pr.is_alive() for pr in procs),new_RL_updates=0,new_native_training_collection=0,old275=None,six54=None,fresh16=None,reload78=None,stable_original_six=False,entered7or8=False,models_not_promoted=True)
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T369 audit900s')));signal.alarm(900);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    obj0=NightLearner(p);data=dataset(p,obj0.actor,lambda:None);contract=contracts(p,obj0,data);objs={};historical=0;events=[json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    for arm in p['arms']:
        obj=NightLearner(p);block=read(OUT/'blocks.json')[arm];assert obj.hashes()==block['input_hashes'];ev=[];trace=fit(p,obj,data,arm,lambda:None,lambda r:ev.append(r));historical+=600
        assert trace==read(OUT/('trace_'+arm+'.json')) and digest(trace)==block['trace_sha256'] and obj.hashes()==block['output_hashes']
        for replica in ['live','shadow']:assert [dict(phase=v[2],step=v[3]) for v in events if v[:2]==[arm,replica]]==ev
        m=s['models'][arm];assert K.C.sha(m['path'])==m['sha256'];z=torch.load(io.BytesIO(gzip.decompress(Path(m['path']).read_bytes())),map_location='cpu',weights_only=False);assert digest(z['state'])==digest(obj.state());objs[arm]=obj
    rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==96 and len(reps)==64;physics=0;full=0
    for r in rows:
        assert K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:assert {k:digest(z[k]) for k in z.files}==r['saved_array_hashes']
        a=arrays(r['path']);assert int(a['native_counts'].astype(int).sum())==r['physics_steps'];physics+=r['physics_steps'];assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout'] and not r['administrative_censor']
        if r['role']=='initial':assert {k:r['array_hashes'][k] for k in p['initial_expected_hashes'][r['job_id']]}==p['initial_expected_hashes'][r['job_id']]
        if not r['full_arrays_saved']:continue
        full+=1;assert {k:digest(v) for k,v in a.items()}==r['array_hashes'];q=obj0.actor if r['role']=='initial' else objs[r['arm']].actor;q.reset_noise();h=None
        with torch.no_grad():
            for k,x in enumerate(a['states']):
                bh=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(bh,a['hidden_before'][k]);action,_,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(action[0].numpy(),a['actions'][k])
                for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k])
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                nn=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);disc=K.R.GAMMA**nn;d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
                rw=-.000002*(1-disc)/(1-K.R.GAMMA)+.03*d0-disc*.03*d1*(not(c or g or t))
                if c or g or t:rw+=K.R.GAMMA**(nn-1)*(float(g)-float(c)-.5*float(t))
                assert abs(rw-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
    for r in reps:
        ref=next(v for v in rows if v['role']=='final_'+r['arm'] and v['job_id']==r['job_id']);assert ref['array_hashes']==r['array_hashes'] and ref['integration_sha256']==r['integration_sha256'] and r['all_arrays_hidden_history_integration_exact'];physics+=r['physics_steps']
    base={k:{v['job_id'] for v in rows[:32][sl] if v['success']} for k,sl in [('old',slice(0,20)),('wide',slice(20,25)),('narrow',slice(27,32))]}
    for arm in p['arms']:
        f=[v for v in rows if v['role']=='final_'+arm];sets={k:{v['job_id'] for v in f[sl] if v['success']} for k,sl in [('old',slice(0,20)),('wide',slice(20,25)),('narrow',slice(27,32))]};assert all(base[k]<=sets[k] for k in base)==s['result'][arm]['retention_pass']
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==163 and s['updates']==dict(live=1200,shadow=1200) and full==3 and s['elapsed_s']<=3600
    result=dict(state='PASS',new_native=0,historical_discarded_Adam=historical,source_model_data_SHA_exact=True,complete_live_shadow_reconstruction=True,contracts=contract,current_head_native_history_reward_exact=True,model_readbacks=64,commands=163,physics_steps=physics,stable_original_six=False)
    save(OUT/'audit.json',result);save(DOC/'T369_AUDIT.json',result);signal.alarm(0)


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();epoch=time.time()
    with (DOC/'T369_DRIVER.log').open('xb') as f:pr=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage matched short-horizon supervision','timeout','--signal=TERM','--kill-after=1s','5400s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=pr.pid,launch_epoch=epoch,deadline_epoch=epoch+3600,all_work_deadline_epoch=epoch+5400,source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')));print(json.dumps(dict(state='LAUNCHED',pid=pr.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);l=read(OUT/'launch.json');active=None
    def call(args,seconds,log):
        nonlocal active
        with log.open('xb') as f:
            active=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),*args],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=active.wait(timeout=seconds)
                if rc:raise subprocess.CalledProcessError(rc,str(args))
            except BaseException:
                try:os.killpg(active.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                if active.poll() is None:
                    try:active.wait(1)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                active.wait();raise
            finally:active=None
    def stop(*_):raise TimeoutError('T369 bounded finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(l['all_work_deadline_epoch']-time.time())-3))
    try:
        save(DOC/'T369_DRIVER.json',dict(pid=os.getpid(),state='NATIVE'));call([str(HERE/'run.py'),'main'],min(3600,l['deadline_epoch']-time.time()),OUT/'main.log');s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        save(DOC/'T369_DRIVER.json',dict(pid=os.getpid(),state='AUDIT'));call([str(HERE/'run.py'),'audit'],min(900,l['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log');assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T369_DRIVER.json',dict(pid=os.getpid(),state='RAW_BACKUP'));call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T369'],min(600,l['all_work_deadline_epoch']-time.time()-20),DOC/'T369_BACKUP.log')
        receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T369.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T369_DRIVER.json',dict(pid=os.getpid(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(receipt)))
    except BaseException:save(DOC/'T369_DRIVER_ERROR.json',dict(error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()
