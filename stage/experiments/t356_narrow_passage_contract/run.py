"""Full-vehicle local narrow-passage evidence; no navigation policy or planner."""
from pathlib import Path
import sys,os,json,copy,math,time,io,signal,shutil,traceback,contextlib,subprocess,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t356_narrow_passage_contract';DOC=ROOT/'stage/docs/narrow_20261105'
spec=importlib.util.spec_from_file_location('stage356_original_native_recorder',ROOT/'stage/experiments/t317_human_demonstration/recorder.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
K,np,torch,mj=R.K,R.np,R.torch,R.mujoco
read=R.read;save=K.C.save;digest=K.S.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t349_consistent_demo'))
from t349_core import ReservedBudget


def robot_collision(m,d,ids):
    hits=[]
    for c in d.contact[:d.ncon]:
        g1,g2=int(c.geom1),int(c.geom2);r1=int(m.geom_bodyid[g1]) in ids;r2=int(m.geom_bodyid[g2]) in ids
        if r1==r2:continue
        other=g2 if r1 else g1;body=int(m.geom_bodyid[other])
        name=((mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,other) or '')+' '+(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,body) or '')).lower()
        if any(v in name for v in ['floor','ground','plane']):continue
        hits.append((float(c.dist),g1,g2))
    hits.sort();return bool(hits and hits[0][0]<=.002),hits


def support(m,d,g,direction):
    t=int(m.geom_type[g]);v=d.geom_xmat[g].reshape(3,3).T@direction;s=m.geom_size[g]
    if t==int(mj.mjtGeom.mjGEOM_BOX):return float(np.abs(v)@s)
    if t==int(mj.mjtGeom.mjGEOM_SPHERE):return float(s[0])
    if t==int(mj.mjtGeom.mjGEOM_CAPSULE):return float(s[0]+s[1]*abs(v[2]))
    if t==int(mj.mjtGeom.mjGEOM_CYLINDER):return float(s[1]*abs(v[2])+s[0]*math.sqrt(max(0.,1-v[2]**2)))
    if t==int(mj.mjtGeom.mjGEOM_ELLIPSOID):return float(np.linalg.norm(s*v))
    raise AssertionError(('Unsupported active collider',g,t))


def active_geoms(env):
    m=env.model
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in env.robot_body_ids and (m.geom_contype[g] or m.geom_conaffinity[g])]


def footprint(env):
    m,d=env.model,env.data;rot=d.xmat[env.base_body_id].reshape(3,3);origin=d.xpos[env.base_body_id];result=[]
    for g in active_geoms(env):
        pos=rot.T@(d.geom_xpos[g]-origin);ext=np.array([support(m,d,g,rot[:,j]) for j in range(3)])
        result.append(dict(id=g,name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),type=int(m.geom_type[g]),
            body=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])),center_body=pos.tolist(),half_extent_body=ext.tolist(),
            min_body=(pos-ext).tolist(),max_body=(pos+ext).tolist(),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])))
    low=np.min([g['min_body'] for g in result],axis=0);high=np.max([g['max_body'] for g in result],axis=0)
    return dict(active_collision_geoms=result,bounds_min=low.tolist(),bounds_max=high.tolist(),outside_span=(high-low).tolist())


def pair_eligible(m,g,h):return bool((int(m.geom_contype[g])&int(m.geom_conaffinity[h])) or (int(m.geom_contype[h])&int(m.geom_conaffinity[g])))


def scene(env,shift):
    m,d=env.model,env.data;before=K.R.integration(env).copy();pos=m.geom_pos.copy();sizes=m.geom_size.copy();masks=(m.geom_contype.copy(),m.geom_conaffinity.copy())
    circle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_circle_obs_9');wall=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
    assert circle>=0 and wall>=0
    if shift:
        m.geom_pos[circle,0]+=shift;mj.mj_setConst(m,d);mj.mj_setState(m,d,before,mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(m,d);mj.mj_setState(m,d,before,mj.mjtState.mjSTATE_INTEGRATION)
    assert np.array_equal(K.R.integration(env),before)
    others=np.arange(m.ngeom)!=circle
    assert np.array_equal(m.geom_pos[others],pos[others]) and np.array_equal(m.geom_size,sizes)
    assert np.array_equal(m.geom_contype,masks[0]) and np.array_equal(m.geom_conaffinity,masks[1])
    gap=float(m.geom_pos[circle,0]-m.geom_size[circle,0]-m.geom_pos[wall,0]-m.geom_size[wall,0])
    return dict(circle_id=int(circle),wall_id=int(wall),gap_m=gap,scene_sha256=K.C.model_hash(env),shift_x_m=shift)


def local_reset(env,p,dx):
    # A separately labelled stationary local diagnostic reset, never an
    # original six-metre trajectory prefix or a navigation success.
    env.data.qpos[:2]=[p['center_x_m']+dx,p['start_y_m']];yaw=-math.pi/2
    env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    env.data.qvel[:]=0;env.data.qacc_warmstart[:]=0;mj.mj_forward(env.model,env.data)
    env.step_count=0;env.prev_action=np.zeros(2,np.float32);env.prev_goal_dist=float(env._goal_state()[1]);env.history.clear()
    obs=np.asarray(env._observation(update_history=False),np.float32)
    assert not env._collision() and not robot_collision(env.model,env.data,env.robot_body_ids)[0]
    return obs


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def prepare():
    assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/experiments/t326_gap_geometry/protocol.json'))
    original=read(ROOT/'stage/runs/t326_gap_geometry/geometry.json');lookup={g['name']:g for g in original['geoms'] if g['name']}
    wall=lookup['real_rect_obs_1'];circle=lookup['real_circle_obs_9'];wallright=wall['world_pos'][0]+wall['size'][0]
    # Fixed gap derives from full neutral tyre span .61m +2*.045m margin.
    targetgap=.70;shift=targetgap-(circle['world_pos'][0]-circle['size'][0]-wallright)
    center=wallright+targetgap/2;start=circle['world_pos'][1]+circle['size'][0]+.43+.05;finish=circle['world_pos'][1]-circle['size'][0]-.43-.05
    p.update(experiment='T356_FULL_COLLIDER_LOCAL_NARROW_PASSAGE',final_bytes=5*1024**2,wall_seconds=900,audit_seconds=300,all_work_seconds=1800,
        max_commands=12,max_physics=96000,local_native_cap=12000,episode_bytes=768*1024,center_x_m=center,start_y_m=start,finish_y_m=finish,
        target_gap_m=targetgap,shift_x_m=shift,static_queries_per_scene=161,action=[.999,0.],
        jobs=[dict(id='original057_center_control',shift=0.,dx=0.),dict(id='narrow070_center',shift=shift,dx=0.),
              dict(id='narrow070_minus1cm',shift=shift,dx=-.01),dict(id='narrow070_plus1cm',shift=shift,dx=.01)],
        preflight_geometry_commands=2,main_geometry_commands=2,
        contract='No Actor/optimizer/planner/navigationtraining. Originalscene.57 unchanged; fullcar activecollider support notjustchassis. Two original/+.13circle9 scenes, all otherparams masks sizes drive5force120 originalstep2ms. Two preflight0nativeenvs+two main0native geometryenvs with161stationary copieddataposes each, allrobotworld collisionpair filters and nativecontact<=.002. Four fixed localstationaryreset straightconstantaction.999,0 probes(originalgap control +new.70center/±.01m), each <=12000native24s +one exactnative replay. All8physics max96000,12envcommands incl4zero,900main300audit1800all/own5MiB/global1GiB/free40GiB. Native successpredicate is wholecar passing circle bottomline,notGOAL. No map/waypoint/teacher fedtoActor; no fabricated expert/onpolicydata. Fulloriginal6start56s stillunproven; localwitnessdoesnotguaranteeallroutes. No width/angle/action/seedgrid or retries, allfailures/censorssaved. Originalmodel/snapshot/sourceimmutable; newscene separatehash, no shrinkingcar/contactmask/timeforce cheats. Phaseisolated.')
    inputs=set(p['input_hashes'])|{f for f in read(ROOT/'stage/experiments/t349_consistent_demo/protocol.json')['input_hashes'] if Path(f).suffix=='.py'}
    inputs|={str(ROOT/'stage/experiments/t349_consistent_demo/t349_core.py'),str(HERE/'run.py'),str(DOC/'T356_PLAN.md'),str(DOC/'FEASIBILITY_SOURCES.md'),str(ROOT/'stage/runs/t326_gap_geometry/geometry.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(inputs)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'];OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',native=0,Adam=0,target_gap=targetgap,shift=shift)))


def preflight():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();torch.set_num_threads(1)
    torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No optimizer'))
    folder=OUT/'preflight';folder.mkdir();geo=[]
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,w,_=R.setup(folder/'native')
        try:
            for shift in [0.,p['shift_x_m']]:
                env,obs,_=K.R.task_env(create,p['job'])
                try:
                    assert K.C.model_hash(env)==read(ROOT/'stage/runs/t326_gap_geometry/geometry.json')['scene_sha256']
                    shape=footprint(env);g=scene(env,shift);local_reset(env,p,0.)
                    assert env.max_wheel==5. and np.array_equal(env.model.actuator_forcerange[[env.id_lr_wheel,env.id_rr_wheel]],np.array([[-120.,120.],[-120.,120.]]))
                    assert env.data.time==0. and env.step_count==0
                    assert abs(shape['outside_span'][1]-.61)<1e-6
                    g.update(whole_vehicle=shape,valid_local_start=True,initial_integration_sha256=K.R.arrays_hash(K.R.integration(env)))
                    geo.append(g)
                finally:w.events.pop(id(env),None);env.close()
        finally:w.close()
    assert geo[0]['gap_m']<geo[0]['whole_vehicle']['outside_span'][1]<geo[1]['gap_m']
    save(OUT/'preflight.json',dict(state='PASS',geometry_commands=2,physics_steps=0,optimizer_calls=0,geometries=geo,
        fullcar_and_tire_masks_active=True,local_reset_only_not_original_start=True,full_capacity_reserved=5*1024**2))
    print(json.dumps(dict(state='PREFLIGHT_PASS',native=0,geometry=2,whole_vehicle_width=geo[0]['whole_vehicle']['outside_span'][1])))


def scan(env,p):
    m=env.model;before=K.R.integration(env).copy();data=mj.MjData(m);mj.mj_copyData(data,m,env.data);q0=data.qpos.copy();g=active_geoms(env)
    obstacles=[j for j in range(m.ngeom) if int(m.geom_bodyid[j]) not in env.robot_body_ids and (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,j) or '').startswith('real_')]
    pairs=[(a,b) for a in g for b in obstacles if pair_eligible(m,a,b)];result=[];segment=np.zeros(6)
    for y in np.linspace(p['start_y_m'],p['finish_y_m'],p['static_queries_per_scene']):
        data.qpos[:]=q0;data.qpos[1]=y;mj.mj_forward(m,data)
        collision,hits=robot_collision(m,data,env.robot_body_ids);nearest=(1.,-1,-1)
        for a,b in pairs:
            dist=float(mj.mj_geomDistance(m,data,a,b,1.,segment))
            if dist<nearest[0]:nearest=(dist,a,b)
        result.append(dict(y_m=float(y),native_contact_collision=collision,minimum_reported_geom_distance_m=nearest[0],distance_pair=list(nearest[1:]),
                           nearest_emitted_contact=hits[0] if hits else None))
    assert np.array_equal(K.R.integration(env),before)
    return dict(queries=len(result),rows=result,any_collision=any(v['native_contact_collision'] for v in result),
        finite_samples_not_continuous_or_global_reachability_proof=True,eligible_robot_world_pairs=len(pairs),
        distance_query_ignores_filters_so_pairs_explicit=True,native_contact_is_authority=True,
        mujoco_version=mj.__version__,disableflags=int(m.opt.disableflags),enableflags=int(m.opt.enableflags),
        ccd_iterations=int(m.opt.ccd_iterations),ccd_tolerance=float(m.opt.ccd_tolerance))


def rollout(p,job,native,create,w,guard,physical):
    env=None;poses=[];times=[];pairs=[];dists=[];collision=False;trunc=False;term=False;steps=0
    try:
        env,obs,_=K.R.task_env(create,p['job']);config=scene(env,job['shift']);local_reset(env,p,job['dx'])
        initial=K.R.integration(env).copy();goal=env.goal_xy.copy();action=np.asarray(p['action'],np.float32);done=False
        while steps<p['local_native_cap']:
            guard();obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;physical()
            collision,hits=robot_collision(env.model,env.data,env.robot_body_ids);assert collision==bool(info.get('collision'))
            poses.append(np.asarray(env._pose()[:3]));times.append(env.data.time);dists.append(hits[0][0] if hits else np.nan);pairs.append(hits[0][1:] if hits else [-1,-1])
            assert config['scene_sha256']==K.C.model_hash(env) and np.array_equal(goal,env.goal_xy)
            done=float(env._pose()[1])<=p['finish_y_m']
            if done or term or trunc:break
        a=dict(native_poses=np.asarray(poses),native_times=np.asarray(times),native_emitted_obstacle_contact_distance=np.asarray(dists),
               native_contact_pairs=np.asarray(pairs,np.int16),constant_action=action,initial_native_integration=initial,final_native_integration=K.R.integration(env).copy())
        row=dict(id=job['id'],job=job,config=config,physics_steps=steps,passed_local_gate=done and not collision,
            collision=collision,local_cap_exhausted=steps==p['local_native_cap'] and not done and not collision,
            administrative_censor=not(done or term or trunc) and steps<p['local_native_cap'],native_goal=bool(info.get('reach_goal')),
            original_start_six_metre_success=False,not_training_or_expert=True,final_pose=list(env._pose()[:3]),
            array_hashes={k:digest(v) for k,v in a.items()},integration_sha256=K.R.arrays_hash(a['final_native_integration']))
        return row,a
    except BaseException:
        if env is not None:
            buff=io.BytesIO();np.savez_compressed(buff,native_poses=np.asarray(poses),native_times=np.asarray(times),final_native_integration=K.R.integration(env))
            b=buff.getvalue();assert len(b)<=p['episode_bytes'];(OUT/'partial.npz').write_bytes(b)
        raise
    finally:
        if env is not None:w.events.pop(id(env),None);env.close()


def main():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'status.json').exists()
    start=time.time();state='FAILED';physics=0;commands=2;rows=[];replays=[];geometry=[];active=None;last_status=0.
    torch.set_num_threads(1);torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No optimizer'))
    meter=ReservedBudget(ROOT/'stage/runs',OUT,p['final_bytes'],p['global_bytes'],p['min_free_bytes'])
    def guard():assert time.time()<start+p['wall_seconds']-15;meter.check()
    def physical():
        nonlocal physics,last_status
        physics+=1;assert physics<=p['max_physics']
        if time.time()-last_status>3:status('LOCAL_NATIVE_OR_REPLAY');last_status=time.time()
    def status(phase):save(OUT/'status.json',dict(state=phase,pid=os.getpid(),updated_epoch=time.time(),main_epoch=start,physics_steps=physics,commands=commands,active=active))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T356main900seconds')));signal.alarm(p['wall_seconds']-2)
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));status('STARTING')
    try:
        with (OUT/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            native,create,w,_=R.setup(OUT/'native')
            try:
                for shift in [0.,p['shift_x_m']]:
                    commands+=1;env,obs,_=K.R.task_env(create,p['job'])
                    try:
                        config=scene(env,shift);shape=footprint(env);local_reset(env,p,0.);status('STATIC_FULL_COLLIDER_SCAN');geometry.append(dict(config=config,footprint=shape,scan=scan(env,p)))
                    finally:w.events.pop(id(env),None);env.close()
                save(OUT/'geometry.json',geometry)
                for job in p['jobs']:
                    commands+=1;active=job['id'];status('LOCAL_NATIVE_STRAIGHT_PROBE');row,a=rollout(p,job,native,create,w,guard,physical)
                    b=io.BytesIO();np.savez_compressed(b,**a);blob=b.getvalue();assert len(blob)<=p['episode_bytes'];path=OUT/'episodes'/(job['id']+'.npz');assert not path.exists();path.write_bytes(blob)
                    row.update(path=str(path),sha256=K.C.sha(path));rows.append(row);save(OUT/'cases.json',rows)
                    commands+=1;status('EXACT_NATIVE_REPLAY');rep,ra=rollout(p,job,native,create,w,guard,physical)
                    assert a.keys()==ra.keys() and all(np.array_equal(v,ra[k],equal_nan=True) for k,v in a.items())
                    assert row['array_hashes']==rep['array_hashes'] and row['integration_sha256']==rep['integration_sha256']
                    replays.append(dict(id=job['id'],physics_steps=rep['physics_steps'],all_arrays_and_native_integration_exact=True,independent_sample=False));save(OUT/'replays.json',replays)
            finally:w.close()
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),physics_steps=physics,commands=commands,active=active))
    signal.alarm(0)
    s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,physics_steps=physics,commands=commands,geometry_commands=4,
        stationary_configuration_queries=sum(g['scan']['queries'] for g in geometry),cases=rows,replays=replays,all_workers_stopped=True,
        Actor_updates=0,optimizer_calls=0,original_six_pass=False,local070_all3_pass=len(rows)==4 and all(v['passed_local_gate'] for v in rows[1:]),
        original057_local_control_pass=bool(rows and rows[0]['passed_local_gate']))
    save(OUT/'summary.json',s);save(OUT/'status.json',s);print(json.dumps(s),flush=True)


def audit():
    p=read(HERE/'protocol.json');verify(p);s=read(OUT/'summary.json');dest=DOC/'T356_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    assert s['commands']==12 and s['Actor_updates']==s['optimizer_calls']==0
    physics=0
    for r,rep in zip(s['cases'],s['replays']):
        assert K.C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:a={k:z[k].copy() for k in z.files}
        assert {k:digest(v) for k,v in a.items()}==r['array_hashes']
        assert len(a['native_poses'])==r['physics_steps'] and np.abs(np.diff(a['native_times'])-.002).max()<1e-10
        assert np.array_equal(a['constant_action'],np.asarray(p['action'],np.float32))
        assert K.R.arrays_hash(a['final_native_integration'])==r['integration_sha256']
        assert rep['all_arrays_and_native_integration_exact'] and not rep['independent_sample'] and rep['physics_steps']==r['physics_steps']
        assert r['passed_local_gate']==(r['final_pose'][1]<=p['finish_y_m'] and not r['collision'])
        assert not r['original_start_six_metre_success'];physics+=r['physics_steps']+rep['physics_steps']
    assert physics==s['physics_steps']<=p['max_physics']
    geoms=read(OUT/'geometry.json');assert sum(g['scan']['queries'] for g in geoms)==322
    assert abs(geoms[0]['footprint']['outside_span'][1]-.61)<1e-6
    assert abs(geoms[0]['config']['gap_m']-.57)<1e-6 and abs(geoms[1]['config']['gap_m']-.70)<1e-6
    save(dest,dict(state='PASS',new_native_steps=0,new_optimizer=0,raw_source_SHA_exact=True,all4_native_replays_exact=True,
        full_vehicle_width_m=geoms[0]['footprint']['outside_span'][1],local070_all3_pass=s['local070_all3_pass'],
        original057_control_pass=s['original057_local_control_pass'],original_six_metre_still_unproven=True,physics_steps=physics,commands=12))
    print(json.dumps(dict(state='AUDIT_PASS',native_added=0,local070_all3_pass=s['local070_all3_pass'])))


def launch():
    p=read(HERE/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as log:proc=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage fullcar narrowlocal passage',
        'timeout','--signal=TERM','--kill-after=1s','900s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=epoch,deadline_epoch=epoch+900,all_work_deadline_epoch=epoch+1800,
        source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print(json.dumps(dict(state='LAUNCHED',pid=proc.pid)))


if __name__=='__main__':globals()[sys.argv[1]]()
