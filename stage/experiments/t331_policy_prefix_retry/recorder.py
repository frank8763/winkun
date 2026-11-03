"""Independent wide-gap human course. Original scene files are read-only.

Only one named cylinder moves +0.60m in world x. Physics, rendering and lidar
share the changed model. Never substitute this course for original acceptance.
"""
from pathlib import Path
import contextlib,importlib.util,json,os,signal,sys,time,copy,math,collections,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t331_policy_prefix_retry';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage331_ui',Path(__file__).with_name('ui_adapter.py'))
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
    assert not OUT.exists()
    p=read(ROOT/'stage/experiments/t328_wide_gap_80s/protocol.json')
    p.update(experiment='T331_POLICY_PREFIX_HUMAN_RECOVERY',port=18770,
        course_id='LEFT6_WIDE80S_BC_PREFIX30S_REAL_HUMAN_CORRECTION',
        prefix_decisions=150,prefix_seconds=30.,max_attempts=1,max_commands=3,
        max_main_commands=2,max_physics=80000,wall_seconds=1200,final_reserve_seconds=240,
        preflight_geometry_commands=1,final_bytes=4*1024**2,payload_bytes=1536*1024,
        metadata_bytes=2*1024**2,episode_bytes=768*1024,
        policy_protocol=str(ROOT/'stage/experiments/t329_success_residual/protocol.json'),
        policy_patch=str(ROOT/'stage/runs/t329_success_residual/bc_proposal.pt.gz'))
    p['contract']='One bounded real learner-prefix then human recovery collection. T329 rejected BC is a research-only sampler, not promoted. Actual mode first150 decisions then exclusive real human control; no reset/goal/clock/history change at handoff. Full80s real native cap, wide1.17m course only; human input limited.999, Actor original mode unchanged within[-1,1]. 1zero-physics geometry+1mixedattempt+1exactreplay/3commands80kphysics1200s last240/4MiB/global1GiB/free40GiB. Prefix actions are not human/expert labels; no invented logp, no Actor fit or RL, not autonomous six success, no7/8, Phaseisolated.'
    files=set(p['input_hashes'])|{str(f) for f in HERE.iterdir() if f.suffix in ['.py','.html','.js']}
    files|={p['policy_protocol'],p['policy_patch'],str(DOC/'T331_PLAN.md'),str(DOC/'T329_AUDIT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T329.json')}
    pp=read(p['policy_protocol']);files|=set(pp['input_hashes'])
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();K.C.save(HERE/'protocol.json',p);K.C.save(OUT/'protocol.json',p)

def prefix_policy(p):
    sys.path.insert(0,str(ROOT/'stage/experiments/t329_success_residual'))
    import t329_learning as L
    q=L.model(read(p['policy_protocol'])).requires_grad_(False)
    patch=L.read_patch(p['policy_patch']);L.load(q,patch['residual'])
    assert K.S.digest(q.state_dict())==patch['policy_hash']
    return q

class App(W.App):
    def __init__(self,p,meter,start,folder):
        self.prefix_actions=[];self.prefix_q=prefix_policy(p)
        super().__init__(p,meter,start,folder)
    def status(self):
        value=super().status();value.update(collection_mode='POLICY_PREFIX_THEN_HUMAN',prefix_seconds=30.,not_autonomous_navigation=True);return value
    def begin(self):
        super().begin();self.state='POLICY_PREFIX';self.write_status()
        for k in range(self.p['prefix_decisions']):
            s=self.session
            x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
            with torch.no_grad():action=self.prefix_q.step(torch.tensor(x[None]))[0][0].numpy()
            s.step(action)
            self.prefix_actions.append(dict(decision=k,action=action.tolist(),state_sha256=K.S.digest(x),source='ACTUAL_T329_BC_POLICY_MODE'))
            self.checkpoint()
            K.C.save(OUT/'prefix_actions.json',self.prefix_actions)
            assert not(s.term or s.trunc),'Prefix ended before planned handoff; keep evidence, no automatic new attempt'
        self.state='HUMAN_RECORDING';self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
        K.C.save(OUT/'handoff.json',dict(decision=len(self.session.rows),steps=self.session.steps,
            scene_sha256=self.session.scene,goal_xy=self.session.goal.tolist(),
            integration_sha256=K.R.arrays_hash(K.R.integration(self.session.env)),
            state_sha256=K.S.digest(K.T.L.packet(self.session.env,self.session.obs,self.session.steps,40000,self.session.previous,self.session.pn)),
            no_reset=True,no_clock_goal_history_change=True))
    def finish_attempt(self,reason):
        super().finish_attempt(reason)
        for row in self.attempts:
            row.update(source='ACTUAL_POLICY_PREFIX_THEN_REAL_HUMAN',human_start_decision=self.p['prefix_decisions'],
                prefix_policy_hash=K.S.digest(self.prefix_q.state_dict()),not_autonomous_navigation=True)
        K.C.save(OUT/'attempts.json',self.attempts)

def preflight():
    p=read(HERE/'protocol.json');R.verify(p);install(p)
    assert not (OUT/'preflight.json').exists()
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
    folder=OUT/'preflight';folder.mkdir();env=None;view=None
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,witness,reward=setup(folder/'native')
        try:
            session=R.Session(p,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero native preflight')))
            env=session.env;q=prefix_policy(p)
            with torch.no_grad():action=q.step(torch.tensor(session.initial120[None]))[0][0].numpy()
            assert action.shape==(2,) and bool((np.abs(action)<=1.000001).all())
            assert session.steps==0 and env.max_steps==40000
            assert K.S.digest(session.initial120)==p['initial_geometry']['initial120_sha256']
            view=R.View(session);image=view.frame(session);assert len(image)>1000
            K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,
                Actor_updates=0,Adam_calls=0,gradient_calls=0,actual_composite_policy_hash=K.S.digest(q.state_dict()),
                first_policy_action=action.tolist(),native_domain_unclipped=True,
                copied_render_native_unchanged=True,strict_initial120_scene_cap=True,
                protocol_sha256=K.C.sha(HERE/'protocol.json')))
        finally:
            if view is not None:view.close()
            if env is not None:witness.events.pop(id(env),None);env.close()
            witness.close()
    print('T331 preflight PASS0physics/Adam,actualcomposite+wide80native+render')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');install(p);start=time.time();meter=app=http=None;state='FAILED';reason='STARTING'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T331 fixed collection window')));signal.alarm(p['wall_seconds']-5)
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
    dest=DOC/'T331_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    prefix=read(OUT/'prefix_actions.json') if (OUT/'prefix_actions.json').exists() else []
    q=prefix_policy(p);torch.set_num_threads(1)
    events=[json.loads(v) for v in (OUT/'request_events.jsonl').read_text().splitlines()] if (OUT/'request_events.jsonl').exists() else []
    inputs=[json.loads(v) for v in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    assert len({v['request_id'] for v in events})==len(events)
    rs=importlib.util.spec_from_file_location('stage331_independent_reward',ROOT/'stage/experiments/t295_hindsight_support/run.py')
    helper=importlib.util.module_from_spec(rs);rs.loader.exec_module(helper)
    physics=0;successful=[]
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes']
        assert row['scene_sha256']==p['initial_geometry']['scene_sha256']!=p['original_geometry']['scene_sha256']
        assert K.R.arrays_hash(initial)==row['initial_integration_sha256']==p['original_geometry']['integration_sha256']
        assert K.R.arrays_hash(final)==row['integration_sha256']
        selected=[v for v in events if v['attempt']==int(row['id'][-2:])]
        offset=p['prefix_decisions']
        assert np.array_equal(np.array([v['action'] for v in selected],np.float32).reshape(-1,2),a['actions'][offset:])
        assert [v['decision'] for v in selected]==list(range(offset,len(a['actions'])))
        human=[v for v in inputs if v['attempt']==int(row['id'][-2:])]
        assert np.array_equal(np.array([v['action'] for v in human],np.float32).reshape(-1,2),a['actions'][offset:])
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps'];physics+=row['physics_steps']
        steps=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]
        assert np.max(np.abs(a['states'][:,112]-(1-steps/40000)))<1e-7 if len(steps) else True
        assert row['physics_steps']<=40000
        if row['timeout']:assert row['physics_steps']==40000
        if len(a['states']):
            assert len(prefix)==p['prefix_decisions'] and len(a['states'])>=len(prefix)
            for k,v in enumerate(prefix):
                with torch.no_grad():expected=q.step(torch.tensor(a['states'][k:k+1]))[0][0].numpy()
                assert np.array_equal(expected,a['actions'][k]) and np.array_equal(expected,np.asarray(v['action'],np.float32))
                assert K.S.digest(a['states'][k])==v['state_sha256']
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
        successful_mixed_policy_human_ids=successful,policy_prefix_actions_verified=True,prefix_not_human_labels=True,original_six_metre_acceptance=False,not_autonomous_navigation_success=True,
        all_original_processes_absent=True,commands=s['commands'],physics_steps=physics,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T331 wide course full raw audit PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
