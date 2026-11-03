"""New bounded human-only collection with telemetry and request deduplication.

The original verified Session/View physics and human-action domain are reused.
No Actor, learned controller, planner, optimizer or automatic physical advance.
"""
from pathlib import Path
import contextlib,importlib.util,json,math,os,re,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t325_human_ui';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage325_original_human',ROOT/'stage/experiments/t317_human_demonstration/recorder.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
R.HERE,R.OUT=HERE,OUT
K,np,torch=R.K,R.np,R.torch
read=R.read

class App(R.App):
    def __init__(self,p,meter,start,folder):
        self.accepted_requests=set();meter.begin(28000)
        super().__init__(p,meter,start,folder);self.geometry_commands=0;self.write_status()
    def status(self):
        x=super().status();s=self.session if self.session is not None else self.preview
        x.update(max_attempts=self.p['max_attempts'],closed=self.closed,decision_index=0 if s is None else len(s.rows),
            remaining_sim_seconds=max(0.,56-x['sim_time_s']),speed_mps=None,front_steer_rad=None,
            commands=self.meter.commands,geometry_commands=1,all_commands=1+self.meter.commands)
        if s is not None:
            e=s.env;pose=e._pose();x['speed_mps']=math.hypot(float(pose[3]),float(pose[4]))
            j=int(e.model.actuator_trnid[e.id_lf_steer,0]);x['front_steer_rad']=float(e.data.qpos[int(e.model.jnt_qposadr[j])])
        elif self.attempts:
            x['sim_time_s']=self.attempts[-1]['physics_steps']*.002;x['remaining_m']=self.attempts[-1]['native_remaining_m']
            x['remaining_sim_seconds']=max(0.,56-x['sim_time_s'])
        return x
    def begin(self):
        self.guard();assert not self.active and not self.closed and self.attempt_number<self.p['max_attempts']
        needed=1 if self.preview is not None else 2
        assert self.meter.commands+needed<=self.p['max_main_commands'] and self.meter.physics+56000<=self.p['max_physics']
        if self.preview is not None:self.session,self.preview=self.preview,None
        else:
            if self.session is not None:self.session.close()
            self.meter.begin(28000);self.session=R.Session(self.p,self.native,self.create,self.witness,self.reward,self.guard,self.physical)
        self.attempt_number+=1;self.current_file=OUT/('attempt_%02d.npz'%self.attempt_number)
        assert not self.current_file.exists();self.active=True;self.state='HUMAN_RECORDING'
        self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
    def step_request(self,data):
        assert self.active and data['expected_decision']==len(self.session.rows)
        request=data['request_id'];assert isinstance(request,str) and re.fullmatch(r'[A-Za-z0-9-]{1,96}',request)
        assert request not in self.accepted_requests;self.accepted_requests.add(request)
        action=R.action_vector(data['action'])
        row=dict(request_id=request,attempt=self.attempt_number,decision=len(self.session.rows),action=action.tolist())
        with (OUT/'request_events.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
        super().step(action)
    def finish_attempt(self,reason):
        super().finish_attempt(reason)
        if any(r['success'] and not r['collision'] for r in self.attempts):self.closed=True
        self.write_status()

class Handler(R.Handler):
    def do_GET(self):
        if self.path=='/control.js':self.send((HERE/'control.js').read_bytes(),'text/javascript; charset=utf-8')
        else:super().do_GET()
    def do_POST(self):
        app=self.server.app
        try:
            assert self.headers.get('Origin')=='http://127.0.0.1:'+str(app.p['port']) and self.headers.get('X-Stage-Control')=='human-ui'
            n=int(self.headers.get('Content-Length','0'));assert 0<=n<=512;data=json.loads(self.rfile.read(n) or b'{}')
            assert self.path in ['/start','/step','/abort','/finish']
            if self.path=='/step':
                R.action_vector(data['action']);assert app.active
                assert type(data['expected_decision']) is int
                assert isinstance(data['request_id'],str) and re.fullmatch(r'[A-Za-z0-9-]{1,96}',data['request_id'])
                if data['expected_decision']!=len(app.session.rows) or data['request_id'] in app.accepted_requests:
                    self.send(json.dumps(dict(error='动作序号已变化或请求已执行；已暂停，请回聊天核对。')).encode(),code=409);return
            elif self.path=='/start':assert not app.active and not app.closed and app.attempt_number<app.p['max_attempts']
        except (AssertionError,ValueError,TypeError,KeyError) as error:
            self.send(json.dumps(dict(error=str(error))).encode(),code=400);return
        try:
            if self.path=='/start':app.begin()
            elif self.path=='/step':app.step_request(data)
            elif self.path=='/abort':
                if app.active:app.finish_attempt('HUMAN_ABORT')
            else:
                if app.active:app.finish_attempt('HUMAN_SESSION_FINISH')
                app.closed=True
            self.send(json.dumps(app.status(),allow_nan=False).encode())
        except BaseException:
            app.failure=traceback.format_exc();app.closed=True
            if app.active:
                try:app.checkpoint()
                except BaseException:pass
            K.C.save(OUT/'http_execution_failure.json',dict(traceback=app.failure,physics_steps=app.meter.physics))
            self.send(json.dumps(dict(error='执行已停止，原始记录已保存，请回聊天检查。')).encode(),code=500)

def prepare():
    assert not OUT.exists();base=read(ROOT/'stage/experiments/t317_human_demonstration/protocol.json')
    p0=read(ROOT/'stage/experiments/t322_full_network_steering/protocol.json')
    files={f for f in p0['input_hashes'] if Path(f).suffix=='.py'}
    files|={str(f) for f in HERE.iterdir() if f.is_file() and f.suffix in ['.py','.js','.html']}
    files|={str(ROOT/v) for v in ['stage/docs/distance_20261103/T325_PLAN.md','stage/docs/distance_20261103/T317_AUDIT.json','stage/docs/long_distance_20261101/release_receipt_T317.json', 'stage/runs/t256_distance_frontier/tasks/base6_LEFT.npz']}
    p=dict(experiment='T325_NEW_HUMAN_TELEMETRY_COLLECTION',job=base['job'],initial_geometry=base['initial_geometry'],
        port=18765,max_attempts=3,max_commands=7,max_main_commands=6,max_physics=168000,wall_seconds=1800,final_reserve_seconds=240,audit_seconds=300,
        episode_bytes=512*1024,preview_bytes=256*1024,final_bytes=4*1024**2,payload_bytes=2*1024**2,metadata_bytes=1536*1024,
        global_bytes=1024**3,min_free_bytes=40*1024**3,input_hashes={f:K.C.sha(f) for f in sorted(files)},
        contract='New user-authorized human-only dataset, up to3attempts stop after first true success, original LEFT6/56simseconds/.20m/collisionpriority/hold100 preserved. NoActor/optimizer/gradient/planner/waypoint/autonomousdriver. Session/View unchanged imports; preview reused as first record. Preflight1zero-stepgeometry+up to3record environments+3exactfull12native replays=7commands/168kphysical. Every physical action from explicit human UI, expected decision/request-id dedup prevents replayed HTTP advance. Pause/slow pacing affects wall only, no reset/threshold/cap changes. Full120/history/actions/counts/poses/rewards/initialfinalintegration retained; all success/failure/censor preserved. Main1800last240replay/closure; no auto old275/six54/fresh16/reload78/7/8 orBC/RL. Own4MiB includes atomicpartial/model-free raw and metadata, global1GiB/free40GiB, single writer. Preflight/client/HTTP/initialhash/render telemetry zero physics, source/inputSHA cleancommit exactremote before unique launch. Endallprocesses before independent raw audit and RAMfullremote eachSHA backup. Phase isolated.')
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    OUT.mkdir();K.C.save(HERE/'protocol.json',p);K.C.save(OUT/'protocol.json',p);print('T325 registered')

def preflight():
    p=read(HERE/'protocol.json');R.verify(p);assert not (OUT/'preflight.json').exists()
    assert read(OUT/'client_contract.json')['state']=='PASS'
    torch.set_num_threads(1);torch.optim.Adam.step=R.U.forbid_step
    folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,witness,reward=R.setup(folder);s=R.Session(p,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('Zero physics')));v=R.View(s)
        try:
            before=K.R.integration(s.env).copy();jpg=v.frame(s);assert len(jpg)<=p['preview_bytes'];(OUT/'preview.jpg').write_bytes(jpg)
            a=App.__new__(App);a.session=s;a.preview=None;a.state='WAITING_FOR_HUMAN';a.active=False;a.closed=False;a.attempt_number=0;a.attempts=[];a.replays=[];a.p=p;a.deadline=time.time()+1800;a.meter=type('Meter',(),dict(physics=0,commands=0))();a.geometry_commands=0
            state=a.status();assert state['decision_index']==0 and state['sim_time_s']==0 and math.isfinite(state['speed_mps']) and math.isfinite(state['front_steer_rad'])
            assert np.array_equal(before,K.R.integration(s.env)) and s.steps==0
        finally:v.close();s.close();witness.close()
    K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        unchanged_original_Session_View=True,telemetry_and_render_do_not_advance_physics=True,initial_geometry_hash_exact=True,protocol_sha256=K.C.sha(HERE/'protocol.json')))
    print('T325 preflight PASS zero physics')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();meter=app=http=None;state='FAILED';reason='STARTING'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T325 fixed collection window')));signal.alarm(p['wall_seconds']-5)
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
        commands=1+(0 if meter is None else meter.commands),geometry_commands=1,physics_steps=0 if meter is None else meter.physics,
        attempts=[] if app is None else app.attempts,replays=[] if app is None else app.replays,
        all_workers_stopped=True,workers_spawned=0,all_renderers_closed=True,Actor_updates=0,Adam_calls=0,gradient_calls=0,
        autonomous_six_metre_pass=False,automatic_training=False,own_bytes=K.S.used(OUT),global_bytes=K.S.used(ROOT/'stage/runs'))
    assert s['commands']<=7 and s['physics_steps']<=168000
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');R.verify(p)
    dest=DOC/'T325_AUDIT.json';assert not dest.exists() and s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc/'+str(pid)).exists()
    events=[json.loads(v) for v in (OUT/'request_events.jsonl').read_text().splitlines()] if (OUT/'request_events.jsonl').exists() else []
    inputs=[json.loads(v) for v in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else []
    assert len({v['request_id'] for v in events})==len(events)
    reward_spec=importlib.util.spec_from_file_location('stage325_independent_reward',ROOT/'stage/experiments/t295_hindsight_support/run.py')
    helper=importlib.util.module_from_spec(reward_spec);reward_spec.loader.exec_module(helper)
    physics=0;successful=[]
    for row in s['attempts']:
        assert K.C.sha(row['path'])==row['sha256'];a,initial,final=R.load_arrays(row['path'])
        assert {k:K.S.digest(v) for k,v in a.items()}==row['array_hashes']
        assert K.R.arrays_hash(initial)==row['initial_integration_sha256']==p['initial_geometry']['integration_sha256']
        assert K.R.arrays_hash(final)==row['integration_sha256']
        selected=[v for v in events if v['attempt']==int(row['id'][-2:])]
        assert np.array_equal(np.array([v['action'] for v in selected],np.float32).reshape(-1,2),a['actions'])
        assert [v['decision'] for v in selected]==list(range(len(a['actions'])))
        human=[v for v in inputs if v['attempt']==int(row['id'][-2:])]
        assert np.array_equal(np.array([v['action'] for v in human],np.float32).reshape(-1,2),a['actions'])
        assert int(a['native_counts'].astype(np.int64).sum())==row['physics_steps'];physics+=row['physics_steps']
        if len(a['states']):
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any()
            for k,x in enumerate(a['states']):
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
                assert abs(helper.reward(d0,d1,int(a['native_counts'][k]),*map(bool,a['end_flags'][k]))-a['rewards'][k])<1e-12
            rep=next(v for v in s['replays'] if v['id']==row['id'])
            assert rep['all12_arrays_exact'] and rep['physics_steps']==row['physics_steps'] and rep['integration_sha256']==row['integration_sha256']
            physics+=rep['physics_steps']
        if row['success'] and not row['collision'] and row['complete']:successful.append(row['id'])
    assert physics==s['physics_steps']<=p['max_physics']
    expected=2 if not s['attempts'] else 1+len(s['attempts'])+len(s['replays'])
    assert s['commands']==expected<=7
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,gradient_calls=0,unique_UI_request_ids=True,request_actions_exact_to_raw=True,
        all_raw120_history_actions_rewards_and_hashes_verified=True,native_replay_receipts_verified=True,
        successful_human_demonstration_ids=successful,not_autonomous_navigation_success=True,
        all_original_processes_absent=True,commands=s['commands'],physics_steps=physics,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T325 full raw + request audit PASS')

if __name__=='__main__':globals()[sys.argv[1]]()
