"""Independent wide-gap human course. Original scene files are read-only.

Only one named cylinder moves +0.60m in world x. Physics, rendering and lidar
share the changed model. Never substitute this course for original acceptance.
"""
from pathlib import Path
import contextlib,importlib.util,json,os,signal,sys,time,copy,math,collections,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t328_wide_gap_80s';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage328_ui',Path(__file__).with_name('ui_adapter.py'))
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
R,K,np,torch=W.R,W.K,W.np,W.torch
W.HERE,W.OUT,R.HERE,R.OUT=HERE,OUT,HERE,OUT
read=R.read
App,Handler=W.App,W.Handler
original_task_env=K.R.task_env
original_setup=R.setup
ACTIVE=None;ACTIVE_WITNESS=None

def setup(folder):
    global ACTIVE_WITNESS
    result=original_setup(folder);ACTIVE_WITNESS=result[2];return result
R.setup=setup

def widen(env,witness):
    p=ACTIVE;m=env.model;before=K.R.integration(env).copy();scene=K.C.model_hash(env)
    witness=ACTIVE_WITNESS if witness is None else witness
    hp,hg,ht=witness.get(env)
    assert scene==p['original_geometry']['scene_sha256']
    gid=R.mujoco.mj_name2id(m,R.mujoco.mjtObj.mjOBJ_GEOM,'real_circle_obs_9')
    assert gid>=0 and int(m.geom_bodyid[gid])==0
    old=m.geom_pos.copy();sizes=m.geom_size.copy();masks=(m.geom_contype.copy(),m.geom_conaffinity.copy())
    m.geom_pos[gid,0]+=p['circle_shift_x_m']
    # Recompute static bounds/BVH rather than leaving broadphase geometry stale.
    R.mujoco.mj_setConst(m,env.data);R.mujoco.mj_setState(m,env.data,before,R.mujoco.mjtState.mjSTATE_INTEGRATION)
    R.mujoco.mj_forward(m,env.data);R.mujoco.mj_setState(m,env.data,before,R.mujoco.mjtState.mjSTATE_INTEGRATION)
    others=np.arange(m.ngeom)!=gid
    assert np.array_equal(m.geom_pos[others],old[others]) and np.array_equal(m.geom_size,sizes)
    assert np.array_equal(m.geom_contype,masks[0]) and np.array_equal(m.geom_conaffinity,masks[1])
    assert np.array_equal(K.R.integration(env),before)
    # Four true factory frames retain their goal/motion/time metadata. Only
    # lidar sector values are refreshed for the new static scene, in-place,
    # keeping Witness object identities and provenance.
    lidar=env._lidar48();frame=env._history_frame(lidar,*env._goal_state()[1:])
    for f in env.history:f[:8]=frame[:8]
    # Restored snapshot frames can be value-matched copies, not original hook
    # objects. Preserve their already-observed metadata before refreshing lidar.
    witness.events[id(env)]=collections.deque([(f,pose.copy(),goal.copy(),float(t))
        for f,pose,goal,t in zip(env.history,hp,hg,ht)],maxlen=48)
    obs=np.asarray(env._observation(update_history=False),np.float32)
    assert not env._collision() and env.step_count==0
    return obs

def task_env(create,job):
    env,obs,actual=original_task_env(create,job)
    try:
        assert K.R.arrays_hash(K.R.integration(env))==ACTIVE['original_geometry']['integration_sha256']
        obs=widen(env,None)
        return env,obs,actual
    except BaseException:env.close();raise

def install(p):
    global ACTIVE
    ACTIVE=p;K.R.task_env=task_env

def prepare():
    assert not OUT.exists();p=read(ROOT/'stage/experiments/t325_human_ui/protocol.json')
    p.update(experiment='T328_WIDE_GAP_HUMAN_COURSE',port=18768,circle_shift_x_m=.60,
        original_geometry=copy.deepcopy(p['initial_geometry']),initial_geometry={},max_attempts=1,max_commands=3,max_main_commands=2,max_physics=80000,wall_seconds=1200,final_reserve_seconds=240,preflight_geometry_commands=1,episode_bytes=768*1024,payload_bytes=1536*1024,
        course_id='LEFT6_CIRCLE9_SHIFT_X_PLUS_0P60_CAP80S',original_acceptance=False)
    files={f for f in p['input_hashes'] if Path(f).suffix=='.py'}
    files|={str(f) for f in HERE.iterdir() if f.suffix in ['.py','.html','.js']}
    files|={str(ROOT/v) for v in ['stage/docs/distance_20261103/T328_PLAN.md','stage/runs/t326_gap_geometry/geometry.json','stage/docs/distance_20261103/T326_AUDIT.json']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    p['job']['case']['step_cap']=40000
    p['job']['id']='wide6_left_cap80s';p['job']['source_reset_id']='wide6_left_cap80s'
    p['contract']='Independent user-authorized wide-gap training course, one cylinder real_circle_obs_9 x+=.60m only, 0.57->1.17m wall gap; original scene files/snapshots/models unmodified. Recompute mj_setConst BVH and forward, preserve native integration; lidar/history sector refresh declared, goal/motion/history metadata retained. Physics/.20m/hold100 unchanged; time limit explicitly80s not original56. New single human attempt+exactreplay/3commands80kphysical/1200s last240/own4MiB/global1GiB/free40GiB. No Actor/RL/planner/waypoint/router. Separate course scenehash and initial120, never original6m success. Preflight1geometry0physics ray/contactoldnew probes/render telemetry, canonical replay later. Raw allsuccessfailurecensor kept and fullremote verified afterallprocesses stopped. Phaseisolated.'
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'];OUT.mkdir()
    K.C.save(HERE/'protocol.json',p);K.C.save(OUT/'protocol.json',p)

def preflight():
    assert not (OUT/'preflight.json').exists();p=read(HERE/'protocol.json');R.verify(p)
    assert read(OUT/'client_contract.json')['state']=='PASS';torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
    folder=OUT/('preflight_recovery_001' if p.get('preflight_geometry_commands')==2 else 'preflight');folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,witness,reward=R.setup(folder);env,obs,_=original_task_env(create,p['job']);view=None
        try:
            assert env.max_steps==40000 and p['job']['case']['step_cap']==40000
            assert '28000' not in (HERE/'session_base.py').read_text()
            clock56=K.T.L.packet(env,obs,28000,40000,np.zeros(2,np.float32),100)
            assert abs(float(clock56[112])-.3)<1e-7
            original=K.R.integration(env).copy();original120=K.T.L.packet(env,obs,0,40000,np.zeros(2,np.float32),0)
            assert K.S.digest(original120)==p['original_geometry']['initial120_sha256']
            install(p);obs=widen(env,witness);x=K.T.L.packet(env,obs,0,40000,np.zeros(2,np.float32),0)
            hp,hg,ht=witness.get(env);assert np.abs(K.H.rebuild(x,np.asarray(env._pose()[:3]),hp,env.goal_xy,hg)-x).max()<=2e-6
            assert np.array_equal(K.R.integration(env),original)
            m=env.model;gid=R.mujoco.mj_name2id(m,R.mujoco.mjtObj.mjOBJ_GEOM,'real_circle_obs_9')
            wall=R.mujoco.mj_name2id(m,R.mujoco.mjtObj.mjOBJ_GEOM,'real_rect_obs_1')
            gap=float(m.geom_pos[gid,0]-m.geom_size[gid,0]-m.geom_pos[wall,0]-m.geom_size[wall,0]);assert abs(gap-1.17)<1e-6
            # mj_ray sees the new static center. Temporary geom groups restored.
            groups=m.geom_group.copy();m.geom_group[:]=0;m.geom_group[gid]=5
            mask=np.zeros(6,np.uint8);mask[5]=1;hit=np.array([-1],np.int32)
            origin=m.geom_pos[gid].copy()+[1.,0.,.11]
            ray=float(R.mujoco.mj_ray(m,env.data,origin,np.array([-1.,0.,0.]),mask,1,-1,hit))
            m.geom_group[:]=groups;assert abs(ray-.75)<1e-6 and int(hit[0])==gid
            # Isolated data probe verifies actual broadphase collision follows
            # the moved cylinder, without stepping or mutating recording state.
            probe=R.mujoco.MjData(m);R.mujoco.mj_setState(m,probe,original,R.mujoco.mjtState.mjSTATE_INTEGRATION)
            probe.qpos[:2]=m.geom_pos[gid,:2];R.mujoco.mj_forward(m,probe)
            contacts=[(int(c.geom1),int(c.geom2)) for c in probe.contact[:probe.ncon]]
            assert any(gid in pair and any(int(m.geom_bodyid[g]) in env.robot_body_ids for g in pair if g!=gid) for pair in contacts)
            assert np.array_equal(K.R.integration(env),original)
            p['initial_geometry']=dict(p['original_geometry'],initial120_sha256=K.S.digest(x),scene_sha256=K.C.model_hash(env),
                original_scene_sha256=p['original_geometry']['scene_sha256'],gap_m=gap)
            K.C.save(OUT/'registration_before_geometry.json',read(HERE/'protocol.json'))
            K.C.save(HERE/'protocol.json',p);K.C.save(OUT/'protocol.json',p)
            fake=type('Session',(),dict(env=env))();view=R.View(fake);jpg=view.frame(fake);assert len(jpg)<=p['preview_bytes'];(OUT/'preview.jpg').write_bytes(jpg)
            assert np.array_equal(K.R.integration(env),original)
            K.C.save(OUT/'scene_change.json',dict(course_id=p['course_id'],geom='real_circle_obs_9',shift_x_m=.6,
                old_center=[.899999976158142,-3.09999990463257,.25],new_center=m.geom_pos[gid].tolist(),gap_m=gap,
                original_scene_sha256=p['original_geometry']['scene_sha256'],course_scene_sha256=p['initial_geometry']['scene_sha256'],
                native_initial_integration_unchanged=True,lidar_ray_at_new_geometry_verified=True,broadphase_contact_at_new_geometry_verified=True,
                all_other_geom_positions_sizes_masks_unchanged=True,original_current_goal_and_history_metadata_preserved=True))
        finally:
            if view is not None:view.close()
            witness.events.pop(id(env),None);env.close();witness.close()
    K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        wide_gap_m=gap,new_scene_geometry_ray_and_contact_verified=True,original_native_initial_state_unchanged=True,
        copied_Session_uses_cap40000_everywhere=True,native_max_steps40000=True,clock_after56_seconds=.3,
        original_View_logic_reused=True,protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T328 wide-gap preflight PASS,0physics')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');install(p);start=time.time();meter=app=http=None;state='FAILED';reason='STARTING'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T328 fixed collection window')));signal.alarm(p['wall_seconds']-5)
    try:
        R.verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
        meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.charge_payload(p['payload_bytes'])
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            app=App(p,meter,start,OUT/'runtime');http=R.HTTPServer(('127.0.0.1',p['port']),Handler);http.timeout=.5;http.app=app
            while not app.closed and time.time()<app.deadline-p['final_reserve_seconds']:http.handle_request();app.write_status()
            if app.failure:raise RuntimeError(app.failure)
            if app.active:app.finish_attempt('ADMINISTRATIVE_COLLECTION_DEADLINE')
        state='COMPLETE';reason='HUMAN_COLLECTION_ENDED' if app.attempts else 'NO_HUMAN_ATTEMPT_EXECUTED'
    except BaseException:K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0 if meter is None else meter.physics))
    finally:
        signal.alarm(0)
        if http is not None:http.server_close()
        if app is not None:app.close()
        if meter is not None:meter.close()
    s=dict(state=state,stop_reason=reason,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,deadline_epoch=start+p['wall_seconds'],
        commands=p['preflight_geometry_commands']+(0 if meter is None else meter.commands),geometry_commands=p['preflight_geometry_commands'],physics_steps=0 if meter is None else meter.physics,
        attempts=[] if app is None else app.attempts,replays=[] if app is None else app.replays,
        all_workers_stopped=True,workers_spawned=0,all_renderers_closed=True,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        autonomous_six_metre_pass=False,original_six_metre_acceptance=False,course_id=p['course_id'],automatic_training=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    assert s['commands']<=p['max_commands'] and s['physics_steps']<=80000
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)
def audit():
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');R.verify(p)
    dest=DOC/'T328_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    events=[json.loads(v) for v in (OUT/'request_events.jsonl').read_text().splitlines()] if (OUT/'request_events.jsonl').exists() else []
    inputs=[json.loads(v) for v in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    assert len({v['request_id'] for v in events})==len(events)
    rs=importlib.util.spec_from_file_location('stage328_independent_reward',ROOT/'stage/experiments/t295_hindsight_support/run.py')
    helper=importlib.util.module_from_spec(rs);rs.loader.exec_module(helper)
    physics=0;successful=[]
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes']
        assert row['scene_sha256']==p['initial_geometry']['scene_sha256']!=p['original_geometry']['scene_sha256']
        assert K.R.arrays_hash(initial)==row['initial_integration_sha256']==p['original_geometry']['integration_sha256']
        assert K.R.arrays_hash(final)==row['integration_sha256']
        selected=[v for v in events if v['attempt']==int(row['id'][-2:])]
        assert np.array_equal(np.array([v['action'] for v in selected],np.float32).reshape(-1,2),a['actions'])
        assert [v['decision'] for v in selected]==list(range(len(a['actions'])))
        human=[v for v in inputs if v['attempt']==int(row['id'][-2:])]
        assert np.array_equal(np.array([v['action'] for v in human],np.float32).reshape(-1,2),a['actions'])
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps'];physics+=row['physics_steps']
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7 if len(steps) else True
        assert row['physics_steps']<=40000
        if row['timeout']:assert row['physics_steps']==40000
        if len(a['states']):
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any()
            for k,x in enumerate(a['states']):
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
                assert abs(helper.reward(d0,d1,int(a['native_counts'][k]),*map(bool,a['end_flags'][k]))-a['rewards'][k])<1e-12
            rep=next(v for v in s['replays'] if v['id']==row['id'])
            assert rep['all12_arrays_exact'] and rep['physics_steps']==row['physics_steps'] and rep['integration_sha256']==row['integration_sha256'];physics+=rep['physics_steps']
        if row['success'] and not row['collision'] and row['complete']:successful.append(row['id'])
    expected=p['preflight_geometry_commands']+(1 if not s['attempts'] else len(s['attempts'])+len(s['replays']))
    assert physics==s['physics_steps']<=p['max_physics'] and s['commands']==expected<=p['max_commands']
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,gradient_calls=0,unique_UI_request_ids=True,request_actions_exact_to_raw=True,
        all_raw120_history_rewards_hashes_verified=True,native_replay_receipts_verified=True,
        successful_wide_course_human_ids=successful,original_six_metre_acceptance=False,not_autonomous_navigation_success=True,
        all_original_processes_absent=True,commands=s['commands'],physics_steps=physics,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T328 wide course full raw audit PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
