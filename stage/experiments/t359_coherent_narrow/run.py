"""One bounded coherent-exploration pilot; no Actor or optimizer update."""
from t359_core import *
import contextlib,ast
import mujoco as mj


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/runs/t357_narrow_origin_transfer/protocol.json'))
    assert read(ROOT/'stage/runs/t358_direct_narrow/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T358.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    rng=np.random.default_rng(20261105359);jobs=[]
    for source_id in [2,4]:
        for k in range(2):
            matrix=rng.normal(size=(16,2));family=f'dev{source_id}_matrix{k}'
            for sign in [1,-1]:
                j=copy.deepcopy(p['jobs'][source_id]);name=f'{family}_{"plus" if sign>0 else "minus"}'
                j.update(id=name,source_reset_id=name,latent=(sign*matrix).tolist(),reset_family=f'narrow070_dev_{source_id}',matrix_family=family,
                    antithetic_sign=sign,diagnostic_group='new070_coherent80');jobs.append(j)
    p.update(experiment='T359_FROZEN_COHERENT_NARROW_EXPLORATION',out=str(OUT),jobs=jobs,seed=20261105359,
        geometry_commands=1,max_commands=17,max_physics=640000,wall_seconds=900,audit_seconds=300,all_work_seconds=2100,
        final_bytes=8*1024**2,full_bytes=768*1024,partial_bytes=768*1024,Actor_updates=0,optimizer_calls=0,
        contract=(DOC/'T359_PLAN.md').read_text())
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T359_PLAN.md'),str(DOC/'T358_FINDINGS.md'),
        str(ROOT/'stage/runs/t358_direct_narrow/audit.json'),str(ROOT/'stage/runs/t358_direct_narrow/cases.jsonl'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T358.json'),str(ROOT/'stage/experiments/t260_distance_archive/backup.py'),str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)};used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T359_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],native_max=640000,commands_max=17))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,independent_latent_families=4,unique_trajectories=8)))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('T359 preflight prohibits optimizers'))
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    q=CoherentActor(p).eval().requires_grad_(False);parent=NightActor(p,0.).eval().requires_grad_(False)
    row=next(json.loads(v) for v in (ROOT/'stage/runs/t358_direct_narrow/cases.jsonl').read_text().splitlines() if json.loads(v)['role'].startswith('train_'))
    a=arrays(row['path']);hidden=other=None;norms=[]
    with torch.no_grad():
        for x in a['states'][:32]:
            xx=torch.tensor(x[None]);action,loc,hidden,_=q.step(xx,hidden);b,mu,other,_=parent.step(xx,other)
            assert torch.equal(action,b) and torch.equal(loc,mu) and torch.equal(hidden,other)
            norms.append(float(np.linalg.norm(q.tape[-1]['features'])))
    assert np.max(np.abs(np.asarray(norms)-1.))<1e-12
    x=torch.tensor(a['states'][0:1]);matrix=np.asarray(p['jobs'][0]['latent']);noises=[]
    with torch.no_grad():
        for sign in [1,-1]:q.latent=sign*matrix;q.reset_noise();q.step(x);noises.append(q.tape[-1]['latent_action_noise'])
    assert np.array_equal(noises[0],-noises[1]) and np.array_equal(matrix,-np.asarray(p['jobs'][1]['latent']))
    assert all(v.grad is None for v in q.parameters()) and digest(q.learned())==digest(parent.learned())
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);module.ACTIVE['circle_shift_x_m']=p['new_scene_shift_x_m'];j=p['jobs'][0]
        try:
            env,obs,_=K.R.task_env(create,j)
            try:
                obs=K.C.perturb_anchor(env,obs,j);g=P.C.B.geometry(env,obs,w,j['case']['step_cap']);m=env.model
                circle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
                gap=float(m.geom_pos[circle,0]-m.geom_size[circle,0]-m.geom_pos[wall,0]-m.geom_size[wall,0])
                assert abs(gap-.70)<1e-6 and g['valid'] and env.data.time==0. and env.step_count==0 and env.max_wheel==5.
                assert np.array_equal(m.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
            finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=1,optimizer_calls=0,zero_latent_parent_exact=True,
        opposite_noise_at_identical_features_exact=True,feature_norm_one=True,fixed_exogenous_feature_variance=[.04,.04],
        not_conditional_rollout_action_density=True,geometry=g,gap_m=gap,capacity=read(DOC/'T359_BUDGET.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,Adam=0,geometry=1)))


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No T357 optimizer'))
    start=time.time();deadline=read(OUT/'launch.json')['deadline_epoch'];ctx=mp.get_context('spawn');counts=ctx.Array('q',2)
    meter=P.C.ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    q=CoherentActor(p).eval().requires_grad_(False);processes=[];pipes=[];rows=[];reps=[];commands=1;active=[];last=0.;phase='STARTING';state='FAILED'
    def status(force=False):
        nonlocal last
        if not force and time.time()-last<3:return
        last=time.time();meter.check();assert time.time()<deadline-5
        save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=last,main_epoch=start,commands=commands,physics_steps=sum(counts[:]),active=active))
    def receive(pipe):
        while not pipe.poll(.25):status()
        return pipe.recv()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T359 deadline')));signal.alarm(max(1,int(deadline-time.time())-3))
    try:
        for k in range(2):
            a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(k+1);proc=ctx.Process(target=worker,args=(b,k,p,counts));proc.start();b.close();processes.append(proc);pipes.append(a)
        os.environ['STAGE_PROCESS_CPU']='0';ready=[receive(pipe) for pipe in pipes];assert all(v['ready'] and v['affinity']==[k+1] for k,v in enumerate(ready))
        save(OUT/'workers.json',dict(parent=os.getpid(),workers=[v.pid for v in processes],ready=ready))
        for offset in range(0,len(p['jobs']),2):
            group=p['jobs'][offset:offset+2];original=[]
            for replica in [False,True]:
                phase='EXACT_REPLAY' if replica else 'COHERENT_EXPLORATION';active=[j['id'] for j in group]
                assert commands+len(group)<=p['max_commands'] and sum(counts[:])+sum(j['case']['step_cap'] for j in group)<=p['max_physics']
                assert time.time()+60<deadline;commands+=len(group);status(True)
                for pipe,j in zip(pipes,group):
                    q.latent=np.asarray(j['latent'],np.float64)
                    pipe.send(dict(j,id=('replay_' if replica else 'primary_')+j['id'],source_reset_id=j['id'],role=phase,arm='memory',
                        policy_hash=q.phash(),deadline_epoch=deadline-15,stochastic=True))
                items=[receive(pipe) for pipe in pipes[:len(group)]];active=[]
                for j,item in zip(group,items):
                    assert 'error' not in item,item.get('error');r,a=item['row'],item['arrays'];q.latent=np.asarray(j['latent'],np.float64);assert r['policy_hash']==q.phash()
                    if not r['complete']:
                        b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(OUT/'partial.npz').write_bytes(blob)
                        save(OUT/'partial_case.json',r);raise RuntimeError('Partial preserved; no retry')
                    r['diagnostic_group']=j['diagnostic_group']
                    if replica:
                        ref=next(v for v in original if v['row']['job_id']==r['job_id'])
                        assert r['array_hashes']==ref['row']['array_hashes'] and r['integration_sha256']==ref['row']['integration_sha256']
                        assert a.keys()==ref['arrays'].keys() and all(np.array_equal(v,ref['arrays'][k]) for k,v in a.items())
                        r.update(all_arrays_hidden_history_integration_exact=True,independent_sample=False);reps.append(r);save(OUT/'replays.json',reps)
                    else:
                        blob=io.BytesIO();np.savez_compressed(blob,**packed(a));data=blob.getvalue();assert len(data)<=p['full_bytes']
                        path=OUT/'episodes'/(j['id']+'.npz');assert not path.exists();path.write_bytes(data)
                        r.update(path=str(path),sha256=K.C.sha(path),full_arrays_saved=True);rows.append(r);original.append(item);save(OUT/'cases.json',rows)
                status(True)
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),active=active,phase=phase,commands=commands,physics_steps=sum(counts[:])))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (OSError,EOFError):pass
        for proc in processes:
            proc.join(5)
            if proc.is_alive():proc.kill();proc.join(3)
        signal.alarm(0)
    result={}
    for name in ['new070_coherent80']:
        rr=[v for v in rows if v['diagnostic_group']==name];result[name]={k:sum(bool(v[k]) for v in rr) for k in ['success','collision','timeout','administrative_censor']};result[name]['executed']=len(rr)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=commands,physics_steps=sum(counts[:]),result=result,
        primary_cases=len(rows),exact_replays=len(reps),Actor_updates=0,optimizer_calls=0,independent_latent_families=4,parent_learned_hash=digest(q.learned()),input_model_sha256=K.C.sha(p['initial_model']),
        all_workers_stopped=all(not v.is_alive() for v in processes),old275=None,six54=None,fresh16=None,reload78=None,stable_six=False,entered7or8=False,budget=meter.stats())
    save(OUT/'summary.json',s);save(OUT/'status.json',s);meter.check(force=True);meter.close();print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T359 audit300s')));signal.alarm(300)
    torch.set_num_threads(1);torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No T359 audit optimizer'))
    rows=read(OUT/'cases.json');reps=read(OUT/'replays.json');assert len(rows)==len(reps)==8
    q=CoherentActor(p).eval().requires_grad_(False);physics=0;coverage=[];matrices={};historical=[];past={}
    for dirname,filename in [('t358_direct_narrow','cases.jsonl'),('t357_narrow_origin_transfer','cases.json')]:
        f=ROOT/'stage/runs'/dirname/filename
        rr=[json.loads(v) for v in f.read_text().splitlines()] if f.suffix=='.jsonl' else read(f)
        for r in rr:
            if not r.get('full_arrays_saved'):continue
            a=arrays(r['path']);h=digest({k:a[k] for k in PHYSICAL_KEYS});past.setdefault(h,[]).append(dict(run=dirname,id=r['id']))
    for r,rep in zip(rows,reps):
        assert K.C.sha(r['path'])==r['sha256'];a=arrays(r['path']);assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
        assert 'logp' not in a and r['per_step_action_logp'] is None and r['latent_density_is_not_action_density']
        q.latent=a['latent_matrix'].copy();q.reset_noise();assert q.phash()==r['policy_hash'] and digest(q.latent)==r['latent_matrix_sha256']
        assert r['latent_matrix_logdensity']==float(-.5*np.square(q.latent).sum()-16*np.log(2*np.pi))
        matrices.setdefault(r['matrix_family'],{})[r['antithetic_sign']]=q.latent.copy();h=None
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]);assert not a['end_flags'][:-1].any() and a['end_flags'][-1].any()
        c,g,t=map(bool,a['end_flags'][-1]);assert c==r['collision'] and (g and not c)==r['success'] and t==r['timeout'] and not r['administrative_censor']
        for k,x in enumerate(a['states']):
            before=np.zeros(16) if h is None else h[0,0].numpy();assert np.array_equal(before,a['hidden_before'][k])
            with torch.no_grad():action,loc,h,_=q.step(torch.tensor(x[None]),h)
            assert np.array_equal(action[0].numpy(),a['actions'][k])
            for name,v in q.tape[-1].items():assert np.array_equal(v,a[name][k]),(r['id'],k,name)
            assert abs(float(np.linalg.norm(a['features'][k]))-1.)<1e-12
            assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
            reward=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))
            if c or g or t:reward+=K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t))
            assert abs(reward-a['rewards'][k])<1e-12
        assert np.array_equal(h[0,0].numpy(),a['last_hidden_after']) and K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps']
        assert rep['job_id']==r['job_id'] and rep['array_hashes']==r['array_hashes'] and rep['integration_sha256']==r['integration_sha256'] and rep['all_arrays_hidden_history_integration_exact']
        assert not rep['independent_sample'];physics+=r['physics_steps']+rep['physics_steps']
        ph=digest({k:a[k] for k in PHYSICAL_KEYS});assert ph==r['physical_digest']
        if ph in past:historical.append(dict(id=r['id'],matches=past[ph],independent_new=False))
        indices=np.flatnonzero(a['poses'][:,1]<=-2.)
        coverage.append(dict(id=r['job_id'],reset_family=r['reset_family'],matrix_family=r['matrix_family'],sign=r['antithetic_sign'],
            first_recorded_pose_y_le_minus2=a['poses'][indices[0]].tolist() if len(indices) else None,
            finite_decision_sample_not_exact_crossing=True,final_pose=a['next_poses'][-1].tolist(),success=r['success']))
    assert len(matrices)==4 and all(np.array_equal(v[1],-v[-1]) for v in matrices.values())
    assert physics==s['physics_steps']<=640000 and s['commands']==17 and s['elapsed_s']<=900 and s['Actor_updates']==s['optimizer_calls']==0
    assert s['parent_learned_hash']==digest(q.learned()) and K.C.sha(p['initial_model'])==s['input_model_sha256']
    result=dict(state='PASS',new_native=0,new_optimizer=0,source_model_all_arrays_exact=True,full120_hidden_history_reward_exact=True,
        latent_matrix_logdensity_only=True,all8_native_regenerations_exact=True,independent_matrix_families=4,unique_physical_trajectories=8,
        historical_physical_matches=historical,coverage=coverage,commands=17,physics_steps=physics,scientific_result=s['result'],learned_navigation_improvement_proven=False)
    save(OUT/'audit.json',result);save(DOC/'T359_AUDIT.json',result);signal.alarm(0);print(json.dumps(dict(state='AUDIT_PASS',native=0,Adam=0)))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T359_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage coherent narrow coverage pilot',
        'timeout','--signal=TERM','--kill-after=1s','2100s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+900,all_work_deadline_epoch=epoch+2100,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T359_DRIVER.json').exists();active=None
    save(DOC/'T359_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='NATIVE'))
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
    def stop(*_):raise TimeoutError('Bounded T359 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'run.py'),'main'],min(900,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T359_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'run.py'),'audit'],min(300,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T359_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T359'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T359_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T359.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T359_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T359.json')))
    except BaseException:save(DOC/'T359_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False))
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

