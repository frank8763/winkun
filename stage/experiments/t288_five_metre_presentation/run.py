"""User-requested HD recordings of two frozen, already accepted five-metre tasks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t288_five_metre_presentation'
sys.path.insert(0,str(ROOT/'stage/experiments/t255_overnight'))
import mem_runtime as M
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
import contextlib,copy,io,json,math,os,shutil,signal,subprocess,time,traceback,types
import cv2,mujoco,imageio_ffmpeg
np,torch,C,R,S=M.np,M.torch,M.C,M.R,M.S


class LimitedLog:
    def __init__(self,path):self.file=Path(path).open('w');self.bytes=0
    def write(self,s):
        s=str(s);left=max(0,65536-self.bytes);b=s.encode()[:left];self.file.write(b.decode(errors='ignore'));self.bytes+=len(b);return len(s)
    def flush(self):self.file.flush()
    def close(self):self.file.close()


def save_blob(path,blob,limit,meter):
    assert not path.exists() and len(blob)<=limit,(path,len(blob),limit)
    meter.charge_payload(len(blob));path.write_bytes(blob)
    return dict(path=str(path),bytes=len(blob),sha256=C.sha(path))


def encoder(path,log_path,width=1920,height=1080,fps=25,max_bytes=14*1024**2):
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','18','-maxrate','1800k','-bufsize','3600k','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-fs',str(max_bytes-65536),str(path)]
    log=Path(log_path).open('wb');process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log);log.close();return process


def verify_movie(path,expected_frames):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened();width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
    assert (width,height)==(1920,1080) and abs(fps-25)<1e-6
    count=0;first=last=None
    while True:
        ok,pic=cap.read()
        if not ok:break
        if first is None:first=pic.copy()
        last=pic;count+=1
    fourcc=int(cap.get(cv2.CAP_PROP_FOURCC));cap.release();assert count==expected_frames,(count,expected_frames)
    assert float(first.std())>15 and float(last.std())>15
    codec=''.join(chr((fourcc>>(8*k))&255) for k in range(4))
    assert codec in ['avc1','h264','H264'],codec
    return dict(width=width,height=height,fps=fps,decoded_frames=count,duration_s=count/fps,codec=codec,all_frames_decoded=True)


class HDView:
    def __init__(self,out,side,plan,meter):
        self.out,self.side,self.plan,self.meter=out,side,plan,meter
        self.movie=out/f'stage_5m_{side.lower()}_1080p.mp4';assert not self.movie.exists()
        meter.charge_payload(plan['movie_bytes']);self.process=encoder(self.movie,out/(side.lower()+'_encoder.log'),max_bytes=plan['movie_bytes'])
        self.frames=0;self.physics=0;self.trace=[];self.ready=False;self.last=None;self.first=None;self.renderers=[];self.posters={}
    def start(self,env):
        self.native_vis=(env.model.vis.global_.offwidth,env.model.vis.global_.offheight)
        self.model=copy.copy(env.model);self.model.vis.global_.offwidth=1280;self.model.vis.global_.offheight=840
        self.model.vis.headlight.ambient[:]=.55;self.model.vis.headlight.diffuse[:]=.65
        assert (env.model.vis.global_.offwidth,env.model.vis.global_.offheight)==self.native_vis
        self.data=mujoco.MjData(self.model);self.renderers=[mujoco.Renderer(self.model,height=840,width=1280),mujoco.Renderer(self.model,height=840,width=640)]
        self.start_xy=np.asarray(env._pose()[:2]).copy();self.goal_xy=env.goal_xy.copy();self.overview=mujoco.MjvCamera();self.detail=mujoco.MjvCamera()
        for cam in [self.overview,self.detail]:mujoco.mjv_defaultCamera(cam)
        self.overview.lookat[:]=[*((self.start_xy+self.goal_xy)/2),.1];self.overview.distance=8.;self.overview.azimuth=90;self.overview.elevation=-78
        self.detail.distance=2.25;self.detail.azimuth=90;self.detail.elevation=-55
        self.ready=True
    def push(self,pic):
        assert self.process.poll() is None,'Encoder stopped before complete verified recording'
        self.process.stdin.write(pic.tobytes());self.frames+=1
    def scene_marks(self,scene):
        for xy,color,size in [(self.start_xy,[.15,.5,1.,1.],.10),(self.goal_xy,[.12,1.,.35,1.],.13)]:
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size]*3),np.array([*xy,.18]),np.eye(3).ravel(),np.asarray(color,dtype=np.float32));scene.ngeom+=1
        points=self.trace[::5]
        for a,b in zip(points[:-1],points[1:]):
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.ones(3)*.008,np.zeros(3),np.eye(3).ravel(),np.array([.1,.65,1.,1.],dtype=np.float32))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.008,np.r_[a,.025],np.r_[b,.025]);scene.ngeom+=1
    def frame(self,env,steps,info,action):
        original=R.integration(env).copy();xy=np.asarray(env._pose()[:2]);self.trace.append(xy.copy());self.detail.lookat[:]=[*xy,.13]
        mujoco.mj_copyData(self.data,self.model,env.data)
        pieces=[]
        for renderer,camera in zip(self.renderers,[self.overview,self.detail]):
            renderer.update_scene(self.data,camera=camera);self.scene_marks(renderer.scene);pieces.append(renderer.render().copy())
        canvas=np.zeros((1080,1920,3),dtype=np.uint8);canvas[:]=[13,22,35];canvas[120:960,:1280]=pieces[0];canvas[120:960,1280:]=pieces[1]
        white=(240,246,252);muted=(164,185,207);green=(60,234,149);blue=(86,190,255)
        cv2.putText(canvas,f'STAGE  |  5 m {self.side} NAVIGATION',(36,52),cv2.FONT_HERSHEY_SIMPLEX,1.15,white,2,cv2.LINE_AA)
        cv2.putText(canvas,'Frozen T255 s2_reset  |  Single neural policy  |  Native MuJoCo physics',(38,92),cv2.FONT_HERSHEY_SIMPLEX,.65,muted,1,cv2.LINE_AA)
        cv2.putText(canvas,'OVERVIEW',(30,154),cv2.FONT_HERSHEY_SIMPLEX,.65,(18,35,50),2,cv2.LINE_AA)
        cv2.putText(canvas,'VEHICLE CLOSE-UP',(1310,154),cv2.FONT_HERSHEY_SIMPLEX,.65,(18,35,50),2,cv2.LINE_AA)
        reached=bool(info.get('reach_goal'));contact=bool(info.get('collision'));label='GOAL REACHED' if reached else 'CONTACT' if contact else 'NAVIGATING'
        remaining=float(env._goal_state()[1]);speed=float(np.linalg.norm(env._pose()[3:5]));color=green if reached else (255,110,110) if contact else blue
        cv2.putText(canvas,label,(38,1010),cv2.FONT_HERSHEY_SIMPLEX,.95,color,2,cv2.LINE_AA)
        cv2.putText(canvas,f'Sim time {steps*.002:6.3f} s    |    Remaining {remaining:.5f} m    |    Speed {speed:.3f} m/s',(450,1007),cv2.FONT_HERSHEY_SIMPLEX,.7,white,1,cv2.LINE_AA)
        cv2.putText(canvas,f'Drive {float(action[0]):+.3f}    Steer {float(action[1]):+.3f}    |    Goal radius 0.20 m    |    Contact: {"YES" if contact else "NO"}',(38,1055),cv2.FONT_HERSHEY_SIMPLEX,.65,muted,1,cv2.LINE_AA)
        assert np.array_equal(R.integration(env),original),'Rendering changed native integration state'
        assert (env.model.vis.global_.offwidth,env.model.vis.global_.offheight)==self.native_vis
        self.last=canvas
        if self.first is None:self.first=canvas.copy()
        return canvas
    def finish(self):
        for _ in range(40):self.push(self.last)
        self.process.stdin.close();assert self.process.wait(timeout=30)==0
        assert self.movie.stat().st_size<=self.plan['movie_bytes']
        qc=verify_movie(self.movie,self.frames)
        for name,pic in [('first',self.first),('goal',self.last)]:
            ok,b=cv2.imencode('.jpg',cv2.cvtColor(pic,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94]);assert ok
            self.posters[name]=save_blob(self.out/f'{self.side.lower()}_{name}.jpg',b.tobytes(),self.plan['poster_bytes'],self.meter)
        return dict(path=str(self.movie),bytes=self.movie.stat().st_size,sha256=C.sha(self.movie),**qc,posters=self.posters,intro_pause_s=.8,goal_pause_s=1.6,nominal_simulation_speed=1.)
    def close(self):
        for renderer in self.renderers:renderer.close()
        if self.process.poll() is None:self.process.kill();self.process.wait()


def main():
    assert not (OUT/'status.json').exists(),'Never restart an ended video directory'
    p=json.loads((HERE/'protocol.json').read_text());start=time.time();meter=None;results=[];physics=0;log=None;encoders=[];state='COMPLETE';active=None
    def alarm(signum,frame):raise TimeoutError('T288 absolute video wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(p['wall_seconds'])
    def status(phase):C.save(OUT/'status.json',dict(state=phase,pid=os.getpid(),main_epoch=start,absolute_deadline_epoch=start+p['wall_seconds'],updated_epoch=time.time(),physics_steps=physics,commands=0 if meter is None else meter.commands,navigation_training_updates=0))
    try:
        assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
        meter=Budget(ROOT,OUT,p,start_epoch=start);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        actor=M.Actor(M.read(p['initial_base'])['policy'],False,2026103501);actor.load_state_dict(M.read(p['checkpoint'])['policy']);actor.eval().requires_grad_(False)
        assert S.digest(actor.state_dict())==p['policy_hash'];C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],encoders=[]));status('READY')
        (OUT/'native').mkdir(exist_ok=True);log=LimitedLog(OUT/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(OUT/'native')
        for side,ref in p['references'].items():
            job=dict(ref['job'],id='hd_'+side.lower(),source_reset_id=ref['source_reset_id']);meter.begin(job['case']['step_cap']);status('RECORDING_'+side)
            active=HDView(OUT,side,p,meter);encoders.append(active.process.pid);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],encoders=encoders));steps=0;last_update=0.
            def checked(env,action):
                nonlocal steps,physics,last_update
                if not active.ready:
                    active.start(env);picture=active.frame(env,0,{},action)
                    for _ in range(20):active.push(picture)
                out=native.step_checked(env,action);steps+=1;physics+=1;meter.physics=physics
                if steps%20==0 or out[2] or out[3]:active.push(active.frame(env,steps,out[4],action))
                if steps%100==0:
                    meter.hold();assert physics<=p['max_physics'] and time.time()<start+p['wall_seconds']-30
                    if time.time()-last_update>=5:status('RECORDING_'+side);last_update=time.time()
                return out
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=M.generate(job,actor,types.SimpleNamespace(step_checked=checked),create,start+p['wall_seconds']-30)
            row=item['row'];buf=io.BytesIO();np.savez_compressed(buf,**item['arrays']);row['payload']=save_blob(OUT/(side.lower()+'_full120.npz'),buf.getvalue(),p['episode_bytes'],meter)
            C.save(OUT/(side.lower()+'_native.json'),row)
            assert row['complete'] and row['success'] and not row['collision'] and not row['timeout']
            assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256'] and row['physics_steps']==ref['physics_steps']
            movie=active.finish();active.close();active=None;results.append(dict(side=side,native=row,movie=movie,all10_arrays_and_integration_exact_to_accepted_baseline=True));C.save(OUT/'recordings.json',results);meter.checkpoint()
        for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    except BaseException:
        state='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,commands=0 if meter is None else meter.commands))
    finally:
        signal.alarm(0)
        if active is not None:active.close()
        if log is not None:log.close()
        if meter is not None:meter.close()
    result=dict(state=state,commands=0 if meter is None else meter.commands,physics_steps=physics,recordings=results,optimizer_calls=0,navigation_training_updates=0,all_workers_stopped=True,all_renderers_and_encoders_closed=True,
        elapsed_s=time.time()-start,source_model_SHA=p['checkpoint_sha256'],policy_hash=p['policy_hash'],own_bytes=S.used(OUT),global_bytes=S.used(ROOT/'stage/runs'),
        limitation='Two user-requested frozen known-task demonstration reproductions. Visual cameras/markers/traces are renderer-only. Not independent new generalization evidence, long-distance success or new algorithm training.')
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
