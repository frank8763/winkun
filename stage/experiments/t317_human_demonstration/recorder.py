"""Bounded, human-operated original LEFT6 recorder. No automatic driver/RL."""
from pathlib import Path
import sys,os,json,io,gzip,time,copy,math,shutil,signal,traceback,contextlib,importlib.util
from http.server import HTTPServer,BaseHTTPRequestHandler
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t316_frozen_native_validation'))
import t316_core as U
np,torch,K=U.np,U.torch,U.K
import mujoco,cv2
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t317_human_demonstration';DOC=ROOT/'stage/docs/distance_20261103'
read=U.read
NAMES=['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards']

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def action_vector(value):
    a=np.asarray(value,dtype=np.float32);assert a.shape==(2,) and np.isfinite(a).all() and bool((np.abs(a)<=.99901).all())
    return a

def setup(folder):
    folder.mkdir(exist_ok=True);native,create=K.S.native_setup(folder)
    spec=importlib.util.spec_from_file_location('stage317_history_witness',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
    return native,create,module.Witness(Base),module.reward

class Session:
    def __init__(self,p,native,create,witness,reward,guard,physical):
        self.p,self.native,self.witness,self.reward,self.guard,self.physical=p,native,witness,reward,guard,physical
        self.env,self.obs,_=K.R.task_env(create,p['job']);self.obs=K.C.perturb_anchor(self.env,self.obs,p['job']);self.steps=0;self.rows=[];self.hist=[];self.previous=np.zeros(2,np.float32);self.pn=0;self.term=self.trunc=False;self.info={}
        assert not self.env._collision();self.initial=K.R.integration(self.env).copy();self.goal=self.env.goal_xy.copy();self.scene=K.C.model_hash(self.env)
        self.initial120=K.T.L.packet(self.env,self.obs,0,28000,self.previous,0);g=p['initial_geometry']
        assert K.S.digest(self.initial120)==g['initial120_sha256'] and K.R.arrays_hash(self.initial)==g['integration_sha256'] and self.scene==g['scene_sha256']
    def step(self,action,count=100):
        self.guard();assert not(self.term or self.trunc);a=action_vector(action);x=K.T.L.packet(self.env,self.obs,self.steps,28000,self.previous,self.pn);pose=np.asarray(self.env._pose()[:3]);meta=self.witness.get(self.env)
        assert np.max(np.abs(K.H.rebuild(x,pose,meta[0],self.goal,meta[1])-x))<=2e-6
        before=self.steps;d0=float(self.env._goal_state()[1])
        for _ in range(min(count,28000-self.steps)):
            self.guard();self.obs,_,self.term,self.trunc,self.info=self.native.step_checked(self.env,a);self.steps+=1;self.physical()
            if self.term or self.trunc:break
        n=self.steps-before;assert 0<n<=100;flags=[bool(self.info.get('collision')),bool(self.info.get('reach_goal')),bool(self.trunc)]
        d1=float(self.env._goal_state()[1]);rw=self.reward(d0,d1,n,*flags);nxt=K.T.L.packet(self.env,self.obs,self.steps,28000,a,n)
        self.rows.append((x,a.copy(),n,nxt,pose,np.asarray(self.env._pose()[:3]),flags,rw));self.hist.append(meta);self.previous=a.copy();self.pn=n
        assert self.scene==K.C.model_hash(self.env) and np.array_equal(self.goal,self.env.goal_xy)
    def arrays(self):
        if self.rows:
            a={k:np.asarray([r[i] for r in self.rows]) for i,k in enumerate(NAMES)};h=self.hist+[self.witness.get(self.env)]
            a.update(history_poses=np.asarray([v[0] for v in h]),history_goals=np.asarray([v[1] for v in h]),history_times=np.asarray([v[2] for v in h]))
        else:
            a=dict(states=np.empty((0,120),np.float32),actions=np.empty((0,2),np.float32),native_counts=np.empty(0,np.uint8),next_states=np.empty((0,120),np.float32),poses=np.empty((0,3)),next_poses=np.empty((0,3)),end_flags=np.empty((0,3),np.uint8),rewards=np.empty(0),history_poses=self.witness.get(self.env)[0][None],history_goals=self.witness.get(self.env)[1][None],history_times=self.witness.get(self.env)[2][None])
        a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8);a['environment_rewards']=a['rewards'].copy()
        assert len(a)==12
        if len(self.rows):assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        return a
    def packet(self):
        a=self.arrays();packed={k:v for k,v in a.items() if k!='next_states'}
        packed['last_next_state']=a['next_states'][-1:] if self.rows else self.initial120[None]
        packed['initial_native_integration']=self.initial;packed['final_native_integration']=K.R.integration(self.env).copy()
        b=io.BytesIO();np.savez_compressed(b,**packed);blob=b.getvalue();assert len(blob)<=self.p['episode_bytes'];return blob,a
    def close(self):
        self.witness.events.pop(id(self.env),None);self.env.close()

class View:
    def __init__(self,session):
        e=session.env;self.model=copy.copy(e.model);self.model.vis.global_.offwidth=960;self.model.vis.global_.offheight=640;self.model.vis.headlight.ambient[:]=.55;self.model.vis.headlight.diffuse[:]=.65
        self.data=mujoco.MjData(self.model);self.renderer=mujoco.Renderer(self.model,height=640,width=960);self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.start=np.asarray(e._pose()[:2]).copy();self.goal=e.goal_xy.copy();self.camera.lookat[:]=[*((self.start+self.goal)/2),.1];self.camera.distance=8.5;self.camera.elevation=-78;self.camera.azimuth=90
    def frame(self,s):
        before=K.R.integration(s.env).copy();mujoco.mj_copyData(self.data,self.model,s.env.data);self.renderer.update_scene(self.data,camera=self.camera);scene=self.renderer.scene
        for xy,color,size in [(self.start,[.15,.5,1.,1.],.08),(self.goal,[.1,1.,.3,1.],.13)]:
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.ones(3)*size,np.r_[xy,.18],np.eye(3).ravel(),np.array(color,np.float32));scene.ngeom+=1
        pose=np.asarray(s.env._pose());g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3)*.015,np.zeros(3),np.eye(3).ravel(),np.array([1.,1.,1.,1.],np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.018,np.r_[pose[:2],.27],np.r_[pose[:2]+.45*np.array([math.cos(pose[2]),math.sin(pose[2])]),.27]);scene.ngeom+=1
        rgb=self.renderer.render().copy();assert np.array_equal(K.R.integration(s.env),before)
        ok,b=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok;return b.tobytes()
    def close(self):self.renderer.close()

def load_arrays(f):
    with np.load(f) as z:packed={k:z[k].copy() for k in z.files}
    initial=packed.pop('initial_native_integration');final=packed.pop('final_native_integration');last=packed.pop('last_next_state')
    packed['next_states']=np.concatenate([packed['states'][1:],last]) if len(packed['states']) else np.empty((0,120),np.float32)
    return packed,initial,final

class App:
    def __init__(self,p,meter,start,folder):
        self.p,self.meter,self.start=p,meter,start;self.deadline=start+p['wall_seconds'];self.native,self.create,self.witness,self.reward=setup(folder)
        self.attempts=[];self.replays=[];self.session=None;self.view=None;self.active=False;self.closed=False;self.attempt_number=0;self.current_file=None;self.in_replay=False;self.failure=None
        self.preview=Session(p,self.native,self.create,self.witness,self.reward,self.guard,self.physical);self.geometry_commands=2;self.view=View(self.preview);self.jpg=self.view.frame(self.preview)
        assert self.preview.steps==0;self.state='WAITING_FOR_HUMAN';self.write_status()
    def guard(self):
        reserve=15 if self.in_replay else self.p['final_reserve_seconds']
        assert time.time()<self.deadline-reserve;self.meter.hold();assert self.meter.physics<self.p['max_physics']
    def physical(self):self.meter.physics+=1
    def status(self):
        s=self.session if self.session is not None else self.preview
        return dict(state=self.state,active=self.active,attempt=self.attempt_number,max_attempts=3,sim_time_s=0 if s is None else .002*s.steps,remaining_m=None if s is None else float(s.env._goal_state()[1]),physics_steps=self.meter.physics,commands=self.geometry_commands+self.meter.commands,deadline_epoch=self.deadline,attempts=self.attempts,replays=self.replays,human_successes=sum(r['success'] and not r['collision'] for r in self.attempts),autonomous_navigation_success=False,Actor_updates=0)
    def write_status(self):
        K.C.save(OUT/'status.json',dict(self.status(),pid=os.getpid(),main_epoch=self.start,updated_epoch=time.time()))
    def checkpoint(self):
        blob,_=self.session.packet();tmp=self.current_file.with_suffix('.partial');tmp.write_bytes(blob);os.replace(tmp,self.current_file)
    def begin(self):
        assert time.time()<self.deadline-self.p['final_reserve_seconds'];self.guard();assert not self.active and self.attempt_number<3
        assert self.geometry_commands+self.meter.commands+2<=self.p['max_commands'] and self.meter.physics+56000<=self.p['max_physics']
        if self.preview is not None:self.preview.close();self.preview=None
        if self.session is not None:self.session.close()
        self.meter.begin(28000);self.session=Session(self.p,self.native,self.create,self.witness,self.reward,self.guard,self.physical);self.attempt_number+=1;self.current_file=OUT/('attempt_%02d.npz'%self.attempt_number);assert not self.current_file.exists();self.active=True;self.state='HUMAN_RECORDING';self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
    def step(self,value):
        self.guard();assert self.active;action=action_vector(value)
        line=json.dumps(dict(attempt=self.attempt_number,decision=len(self.session.rows),action=action.tolist(),epoch=round(time.time(),3)),separators=(',',':'))+'\n';assert len(line.encode())<=128
        with (OUT/'human_inputs.jsonl').open('a') as f:f.write(line);f.flush()
        self.session.step(action);self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
        if self.session.term or self.session.trunc:self.finish_attempt('NATIVE_TERMINAL')
    def finish_attempt(self,reason):
        assert self.active;self.active=False;self.checkpoint();s=self.session;arrays=s.arrays();_,_,final=load_arrays(self.current_file)
        row=dict(id='attempt_%02d'%self.attempt_number,path=str(self.current_file),sha256=K.C.sha(self.current_file),physics_steps=s.steps,decisions=len(s.rows),source='HUMAN_BROWSER_EXPLICIT_ACTIONS',
            complete=bool(s.term or s.trunc),success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),collision=bool(s.info.get('collision')),timeout=bool(s.trunc),administrative_censor=not bool(s.term or s.trunc),stop_reason=reason,
            goal_xy=s.goal.tolist(),scene_sha256=s.scene,initial120_sha256=K.S.digest(s.initial120),initial_integration_sha256=K.R.arrays_hash(s.initial),integration_sha256=K.R.arrays_hash(final),array_hashes={k:K.S.digest(v) for k,v in arrays.items()},native_remaining_m=float(s.env._goal_state()[1]),not_current_Actor_onpolicy=True,no_expert_action_logp_invented=True)
        self.attempts.append(row);K.C.save(OUT/'attempts.json',self.attempts);self.state='EXACT_NATIVE_ACTION_REPLAY';self.write_status()
        if s.rows and time.time()<self.deadline-15:
            self.in_replay=True
            self.meter.begin(28000);s.close();self.session=None;r=Session(self.p,self.native,self.create,self.witness,self.reward,self.guard,self.physical)
            try:
                for action,n in zip(arrays['actions'],arrays['native_counts']):r.step(action,int(n))
                recovered=r.arrays();assert recovered.keys()==arrays.keys() and all(np.array_equal(v,recovered[k]) for k,v in arrays.items())
                assert K.R.arrays_hash(K.R.integration(r.env))==row['integration_sha256']
                self.replays.append(dict(id=row['id'],physics_steps=r.steps,all12_arrays_exact=True,integration_sha256=row['integration_sha256'],not_independent_navigation_sample=True));K.C.save(OUT/'replays.json',self.replays)
            finally:r.close();self.in_replay=False
        self.state='ATTEMPT_SAVED';self.meter.checkpoint();self.write_status()
        if self.attempt_number>=3:self.closed=True
    def close(self):
        if self.session is not None:self.session.close();self.session=None
        if self.preview is not None:self.preview.close();self.preview=None
        self.view.close();self.witness.close()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,body,kind='application/json',code=200):
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        app=self.server.app
        if self.path=='/':self.send((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path.startswith('/frame.jpg'):self.send(app.jpg,'image/jpeg')
        elif self.path=='/status':self.send(json.dumps(app.status(),allow_nan=False).encode())
        else:self.send(b'{}',code=404)
    def do_POST(self):
        app=self.server.app
        try:
            assert self.headers.get('Origin')=='http://127.0.0.1:'+str(app.p['port']) and self.headers.get('X-Stage-Control')=='human-ui'
            length=int(self.headers.get('Content-Length','0'));assert 0<=length<=512;data=json.loads(self.rfile.read(length) or b'{}')
            assert self.path in ['/start','/step','/abort','/finish']
            if self.path=='/step':
                action_vector(data['action']);assert app.active,'No active human attempt'
            if self.path=='/start':assert not app.active and app.attempt_number<3,'No remaining attempt'
        except (AssertionError,ValueError,KeyError,TypeError) as error:
            self.send(json.dumps(dict(error=str(error))).encode(),code=400);return
        try:
            if self.path=='/start':app.begin()
            elif self.path=='/step':app.step(data['action'])
            elif self.path=='/abort':
                if app.active:app.finish_attempt('HUMAN_ABORT')
            elif self.path=='/finish':
                if app.active:app.finish_attempt('HUMAN_SESSION_FINISH')
                app.closed=True
            self.send(json.dumps(app.status(),allow_nan=False).encode())
        except BaseException:
            app.failure=traceback.format_exc();app.closed=True
            if app.active:
                try:app.checkpoint()
                except BaseException:pass
            K.C.save(OUT/'http_execution_failure.json',dict(traceback=app.failure,physics_steps=app.meter.physics,administrative_failure_not_navigation_success=True))
            self.send(json.dumps(dict(error='执行已停止；原始证据已保留，请回聊天检查。')).encode(),code=500)

def preflight():
    assert not any((OUT/n).exists() for n in ['preflight.json','launch.json','status.json']);p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);torch.optim.Adam.step=U.forbid_step;folder=OUT/'preflight';folder.mkdir()
    with (folder/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,witness,reward=setup(folder);s=Session(p,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('No preflight physics')));v=View(s)
        try:
            before=K.R.integration(s.env).copy();jpg=v.frame(s);assert len(jpg)<=p['preview_bytes'];(OUT/'preview.jpg').write_bytes(jpg);assert np.array_equal(K.R.integration(s.env),before) and s.steps==0
            blob,arrays=s.packet();assert len(blob)<=p['episode_bytes']
            for bad in [[1.,0.],[float('nan'),0.],[0.]]:
                try:action_vector(bad)
                except (AssertionError,ValueError):continue
                raise AssertionError('Bad action accepted')
            assert len((HERE/'index.html').read_bytes())<32*1024
        finally:v.close();s.close();witness.close()
    K.C.save(OUT/'preflight.json',dict(state='PASS',geometry_commands=1,physics_steps=0,Adam_calls=0,gradient_calls=0,recorded_human_actions=0,actual_original_LEFT6_initial_hash_exact=True,renderer_copy_did_not_mutate_native=True,JPEG_bytes=len(jpg),open_interval_human_actions_enforced=True,no_expert_or_Actor_logp_created=True,protocol_sha256=K.C.sha(HERE/'protocol.json'),full_worstcase_reserved_bytes=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']))
    print('T317 demonstration preflight PASS,0physics/Adam/gradient/humanActions')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');start=time.time();state='FAILED';reason='STARTING';meter=None;app=None;http=None
    def alarm(*_):raise TimeoutError('T317 bounded human collection window expired')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds']-5)
    try:
        verify(p);assert read(OUT/'preflight.json')['state']=='PASS';torch.set_num_threads(1);torch.optim.Adam.step=U.forbid_step;meter=K.Budget(ROOT,OUT,p,start_epoch=start);meter.charge_payload(p['payload_bytes'])
        with (OUT/'native_factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            app=App(p,meter,start,OUT/'runtime');http=HTTPServer(('127.0.0.1',p['port']),Handler);http.timeout=.5;http.app=app
            while not app.closed and time.time()<app.deadline-p['final_reserve_seconds']:
                http.handle_request();app.write_status()
            if app.failure is not None:raise RuntimeError(app.failure)
            if app.active:app.finish_attempt('ADMINISTRATIVE_COLLECTION_DEADLINE')
        state='COMPLETE';reason='HUMAN_COLLECTION_ENDED' if app.attempts else 'NO_HUMAN_ATTEMPT_EXECUTED'
    except BaseException:
        K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=0 if meter is None else meter.physics,not_autonomous_navigation_result=True))
        if app is not None and app.active:
            try:app.checkpoint()
            except BaseException:pass
    finally:
        signal.alarm(0)
        if http is not None:http.server_close()
        if app is not None:app.close()
        if meter is not None:meter.close()
    s=dict(state=state,stop_reason=reason,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=1 if app is None else app.geometry_commands+meter.commands,geometry_commands=1 if app is None else 2,physics_steps=0 if meter is None else meter.physics,
        attempts=[] if app is None else app.attempts,replays=[] if app is None else app.replays,all_workers_stopped=True,workers_spawned=0,all_renderers_closed=True,Adam_calls=0,gradient_calls=0,Actor_updates=0,autonomous_six_metre_pass=False,automatic_training=False,global_bytes=K.S.used(ROOT/'stage/runs'),own_bytes=K.S.used(OUT))
    K.C.save(OUT/'summary.json',s);K.C.save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    dest=DOC/'T317_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE';verify(p)
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Process still exists',pid))
    inputs=[json.loads(x) for x in (OUT/'human_inputs.jsonl').read_text().splitlines()] if (OUT/'human_inputs.jsonl').exists() else [];physics=0;successful=[]
    for r in s['attempts']:
        assert K.C.sha(r['path'])==r['sha256'];a,initial,final=load_arrays(r['path']);assert {k:K.S.digest(v) for k,v in a.items()}==r['array_hashes']
        assert K.R.arrays_hash(initial)==r['initial_integration_sha256']==p['initial_geometry']['integration_sha256'] and K.R.arrays_hash(final)==r['integration_sha256']
        assert int(a['native_counts'].astype(np.int64).sum())==r['physics_steps'];physics+=r['physics_steps']
        ii=[v for v in inputs if v['attempt']==int(r['id'].split('_')[-1])];assert len(ii)==len(a['actions']) and np.array_equal(np.asarray([x['action'] for x in ii],np.float32).reshape(-1,2),a['actions'])
        if len(a['states']):
            assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['end_flags'][:-1].any()
            for k,x in enumerate(a['states']):
                assert np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],r['goal_xy'],a['history_goals'][k])-x).max()<=2e-6
                flags=a['end_flags'][k];d0=K.H.fields(a['poses'][k],r['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],r['goal_xy'])[1]
                spec=importlib.util.spec_from_file_location('stage317_reward_audit',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                rw=module.reward(d0,d1,int(a['native_counts'][k]),*map(bool,flags));assert abs(rw-a['rewards'][k])<1e-12
            rep=next(v for v in s['replays'] if v['id']==r['id']);assert rep['all12_arrays_exact'] and rep['integration_sha256']==r['integration_sha256'] and rep['physics_steps']==r['physics_steps'];physics+=rep['physics_steps']
        if r['success'] and not r['collision'] and r['complete']:successful.append(r['id'])
    assert physics==s['physics_steps']<=p['max_physics'];assert s['commands']==2+len(s['attempts'])+len(s['replays'])<=p['max_commands']
    K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,gradient_calls=0,all_raw120_history_human_actions_rewards_and_hashes_exact=True,existing_native_action_replay_receipts_verified=True,
        successful_human_demonstration_ids=successful,not_autonomous_navigation_success=True,all_original_processes_absent=True,physics_steps=physics,commands=s['commands'],summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T317 raw human demonstration audit PASS; not autonomous success')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);a=parser.parse_args();globals()[a.mode]()
