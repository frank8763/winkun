from t367_core import *
import contextlib,ast
import mujoco as mj


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists()
    assert read(DOC/'T366_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE'
    assert read(ROOT/'stage/runs/t366_factual_history/audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T366.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/runs/t365_correction_aggregation/protocol.json'))
    source=read(ROOT/'stage/runs/t365_correction_aggregation/summary.json');prior=read(ROOT/'stage/runs/t365_correction_aggregation/cases.json')
    factual=read(ROOT/'stage/runs/t366_factual_history/cases.json');assert len(factual)==7
    p.update(id='T367',folder=OUT.name,out=str(OUT),experiment='T367_FRESH_CENTRE_PPO_FACTUAL_HISTORY_REHEARSAL',initial_model=source['model']['path'],initial_is_T349=False,
        initial_expected_digests={v['job_id']:digest(v['array_hashes']) for v in prior if v['role']=='final'},
        factual_rehearsal=factual,arms={'iid':0.},credit='full_G0',curriculum=False,rounds=2,epochs=4,actor_lr=.0001,
        max_train_episodes=8,max_full_cases=8,max_compact_cases=128,max_commands=179,max_physics=7160000,
        wall_seconds=3600,final_reserve_seconds=1200,audit_seconds=900,all_work_seconds=5400,
        final_bytes=12*1024**2,full_bytes=512*1024,compact_bytes=8*1024,partial_bytes=512*1024,model_bytes=512*1024,
        metadata_bytes=2*1024**2,logs_bytes=256*1024,geometry_commands=3,preflight_discarded_Adam=16,audit_historical_discarded_Adam=8,
        max_live_Adam=8,max_shadow_Adam=8,new_RL_updates=8,seed=20261105367,contract=(DOC/'T367_PLAN.md').read_text())
    worst=8*p['full_bytes']+128*p['compact_bytes']+2*p['model_bytes']+3*p['partial_bytes']+p['metadata_bytes']+p['logs_bytes']
    used=K.S.used(ROOT/'stage/runs');assert worst<p['final_bytes'] and used+p['final_bytes']<p['global_bytes']
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T367_PLAN.md'),str(DOC/'T366_CLOSEOUT.json'),str(DOC/'T366_FINDINGS.md'),str(ROOT/'stage/runs/t366_factual_history/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T366.json'),p['initial_model'],str(ROOT/'stage/runs/t357_narrow_origin_transfer/episodes/narrow070_dev_0.npz')}
    for r in factual:files|={r['path']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T367_BUDGET.json',dict(state='FULL_RESERVATION_PASS',global_before=used,own_max=p['final_bytes'],worst_payload=worst,global_full_reservation=used+p['final_bytes'],global_limit=p['global_bytes'],max_commands=179,max_physics=7160000,live_Adam=8,shadow_Adam=8,preflight_discarded_Adam=16,historical_audit_discarded_Adam=8))
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,full_global_reservation=used+p['final_bytes'])))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    obj=NightLearner(p,0.);initial=obj.state();refs=references(p,obj.actor)
    for g in refs[2]:assert float((obj.actor.sequence(g['x'],g['b']).detach()-g['y']).abs().max())<1e-10
    mask_report=mask_contract(obj,(refs[0],refs[1],refs[2]))
    path=ROOT/'stage/runs/t357_narrow_origin_transfer/episodes/narrow070_dev_0.npz';source=arrays(path);items=[]
    for j in range(4):
        # Synthetic sampler tape on fixed historical observations. These are
        # zero-native numerical tests, never on-policy navigation experience.
        x=source['states'][:24].copy();q=NightActor(p,0.).requires_grad_(False);q.sample=True;q.rng=np.random.default_rng(35800+j);q.reset_noise();h=None
        with torch.no_grad():
            for row in x:_,_,h,_=q.step(torch.tensor(row[None]),h)
        a=dict(states=x,native_counts=np.full(len(x),100),rewards=np.full(len(x),-.001),raw_actions=np.asarray([v['raw_actions'] for v in q.tape]),
            locations=np.asarray([v['locations'] for v in q.tape]),logp=np.asarray([v['logp'] for v in q.tape]));a['rewards'][-1]=[1.,-.5,-1.,.2][j]
        items.append(dict(arrays=a))
    calls=0
    def event(rnd,replica,r):
        with (OUT/'preflight_Adam_events.jsonl').open('a') as f:f.write(json.dumps([rnd,replica,r['phase'],r['step']])+ '\n')
    for rnd in range(2):
        before=obj.state();immutable=digest(before);shadow=NightLearner(p,0.);shadow.load(before)
        live=optimize(p,obj,items,refs,rnd,lambda:None,lambda r:event(rnd,'live',r));calls+=4
        other=optimize(p,shadow,items,refs,rnd,lambda:None,lambda r:event(rnd,'shadow',r));calls+=4
        assert live==other and obj.hashes()==shadow.hashes() and digest(before)==immutable
        if rnd==0:
            # A new matched synthetic tape must use the changed policy density
            # for the mature-Adam numerical test, without overwriting logp.
            mature=obj.state()
            for j,item in enumerate(items):
                q=NightActor(p,0.).requires_grad_(False);q.load_learned(obj.actor.learned());q.sample=True;q.rng=np.random.default_rng(35900+j);q.reset_noise();h=None
                with torch.no_grad():
                    for row in item['arrays']['states']:_,_,h,_=q.step(torch.tensor(row[None]),h)
                for name in ['raw_actions','locations','logp']:item['arrays'][name]=np.asarray([v[name] for v in q.tape])
        else:
            obj.load(before);assert digest(obj.state())==immutable and digest(before)==digest(mature)
    obj.load(initial);assert digest(obj.state())==digest(initial) and calls==16
    folder=OUT/'preflight';folder.mkdir();geos=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE)
        try:
            for j,gap in zip([p['narrow_jobs'][0],p['course_dev_jobs'][0],p['original_jobs'][0]],[.70,1.17,.57]):
                shift=p['training_shift_x_m'] if j['id'].startswith('narrow070') else .6
                module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=shift
                env,obs,_=K.R.task_env(create,j)
                try:
                    obs=K.C.perturb_anchor(env,obs,j);g=P.C.B.geometry(env,obs,w,j['case']['step_cap'])
                    c=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
                    actual=float(env.model.geom_pos[c,0]-env.model.geom_size[c,0]-env.model.geom_pos[wall,0]-env.model.geom_size[wall,0])
                    assert abs(actual-gap)<1e-6 and g['valid'] and env.data.time==0. and env.step_count==0 and env.max_wheel==5.
                    assert np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
                    geos.append(dict(id=j['id'],gap_m=actual,geometry=g))
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,geometry_commands=3,discarded_synthetic_Adam=calls,synthetic_tapes_not_navigation=True,human_prefix20_masked=True,
        mature_Adam_shadow_rollback_exact=True,initial_narrow_support_anchor_error_below_1e_minus10=True,mask_contract=mask_report,wide_targets_from_factual_T353=True,geometries=geos,global_reservation=read(DOC/'T367_BUDGET.json')))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=3,discarded_Adam=calls)))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (DOC/'T367_DRIVER.log').open('xb') as f:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage narrow human demonstration learning',
        'timeout','--signal=TERM','--kill-after=1s','5400s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'driver'],cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+3600,all_work_deadline_epoch=epoch+5400,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


def driver():
    p=read(HERE/'protocol.json');verify(p);launch=read(OUT/'launch.json');assert not (DOC/'T367_DRIVER.json').exists();active=None
    save(DOC/'T367_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='NATIVE'))
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
    def stop(*_):raise TimeoutError('Bounded T367 finalizer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(launch['all_work_deadline_epoch']-time.time())-3))
    try:
        call([str(HERE/'phase.py'),str(OUT)],min(3600,launch['deadline_epoch']-time.time()),OUT/'main.log')
        s=read(OUT/'summary.json')
        for pid in [s['pid'],*read(OUT/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
        assert s['state']=='COMPLETE','Raw failure preserved; no navigation restart'
        save(DOC/'T367_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='INDEPENDENT_AUDIT'))
        call([str(HERE/'audit.py'),str(OUT)],min(900,launch['all_work_deadline_epoch']-time.time()-30),OUT/'audit_console.log')
        assert read(OUT/'audit.json')['state']=='PASS'
        save(DOC/'T367_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='RAW_BACKUP'))
        call([str(ROOT/'stage/experiments/t260_distance_archive/small_parts.py'),'T367'],min(600,launch['all_work_deadline_epoch']-time.time()-20),DOC/'T367_BACKUP.log')
        receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T367.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        save(DOC/'T367_DRIVER.json',dict(pid=os.getpid(),epoch=time.time(),state='COMPLETE_AUDIT_AND_RAW_BACKUP',summary_sha256=K.C.sha(OUT/'summary.json'),receipt_sha256=K.C.sha(ROOT/'stage/docs/long_distance_20261101/release_receipt_T367.json')))
    except BaseException:
        save(DOC/'T367_DRIVER_ERROR.json',dict(epoch=time.time(),error=traceback.format_exc(),navigation_restarted=False));raise
    finally:signal.alarm(0)


if __name__=='__main__':globals()[sys.argv[1]]()

