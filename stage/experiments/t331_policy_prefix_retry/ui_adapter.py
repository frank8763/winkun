"""New bounded human-only collection with telemetry and request deduplication.

The original verified Session/View physics and human-action domain are reused.
No Actor, learned controller, planner, optimizer or automatic physical advance.
"""
from pathlib import Path
import contextlib,importlib.util,json,math,os,re,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t331_policy_prefix_retry';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage331_original_human',Path(__file__).with_name('session_base.py'))
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
R.HERE,R.OUT=HERE,OUT
K,np,torch=R.K,R.np,R.torch
read=R.read

def human_action(value):
    a=R.action_vector(value)
    assert bool((np.abs(a)<=.99901).all())
    return a

class App(R.App):
    def __init__(self,p,meter,start,folder):
        self.accepted_requests=set();meter.begin(40000)
        super().__init__(p,meter,start,folder);self.geometry_commands=0;self.write_status()
    def status(self):
        x=super().status();s=self.session if self.session is not None else self.preview
        x.update(max_attempts=self.p['max_attempts'],closed=self.closed,decision_index=0 if s is None else len(s.rows),
            remaining_sim_seconds=max(0.,80-x['sim_time_s']),speed_mps=None,front_steer_rad=None,
            commands=self.meter.commands,geometry_commands=1,all_commands=1+self.meter.commands)
        if s is not None:
            e=s.env;pose=e._pose();x['speed_mps']=math.hypot(float(pose[3]),float(pose[4]))
            j=int(e.model.actuator_trnid[e.id_lf_steer,0]);x['front_steer_rad']=float(e.data.qpos[int(e.model.jnt_qposadr[j])])
        elif self.attempts:
            x['sim_time_s']=self.attempts[-1]['physics_steps']*.002;x['remaining_m']=self.attempts[-1]['native_remaining_m']
            x['remaining_sim_seconds']=max(0.,80-x['sim_time_s'])
        return x
    def begin(self):
        self.guard();assert not self.active and not self.closed and self.attempt_number<self.p['max_attempts']
        needed=1 if self.preview is not None else 2
        assert self.meter.commands+needed<=self.p['max_main_commands'] and self.meter.physics+80000<=self.p['max_physics']
        if self.preview is not None:self.session,self.preview=self.preview,None
        else:
            if self.session is not None:self.session.close()
            self.meter.begin(40000);self.session=R.Session(self.p,self.native,self.create,self.witness,self.reward,self.guard,self.physical)
        self.attempt_number+=1;self.current_file=OUT/('attempt_%02d.npz'%self.attempt_number)
        assert not self.current_file.exists();self.active=True;self.state='HUMAN_RECORDING'
        self.checkpoint();self.jpg=self.view.frame(self.session);self.write_status()
    def step_request(self,data):
        assert self.active and data['expected_decision']==len(self.session.rows)
        request=data['request_id'];assert isinstance(request,str) and re.fullmatch(r'[A-Za-z0-9-]{1,96}',request)
        assert request not in self.accepted_requests;self.accepted_requests.add(request)
        action=human_action(data['action'])
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
                human_action(data['action']);assert app.active
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

