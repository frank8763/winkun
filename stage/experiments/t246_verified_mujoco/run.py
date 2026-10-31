"""Fresh8centerMuJoCoruns, physics-exact and budgeted preview only after gate."""
from pathlib import Path
import sys,os,io,gzip,json,time,traceback,subprocess,shutil,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t246_verified_mujoco'
sys.path.insert(0,str(ROOT/'stage/experiments/t219_rnd_runtime'))
import support as S
C,R,np,torch=S.C,S.R,S.np,S.torch
import mujoco,cv2,imageio_ffmpeg

def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicateT246'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();cases=[];physics=0;encoder=None;reason=None;frame_count=0;verified=[]
    def guard():
        assert time.monotonic()-start<plan['wall_seconds'],'WALL_LIMIT'
        assert S.used(OUT)<plan['final_bytes'] and S.used(ROOT/'stage/runs')<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3,'STORAGE_LIMIT'
        assert physics<=plan['max_physics']
    try:
        for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        state=torch.load(io.BytesIO(gzip.decompress(Path(plan['checkpoint']).read_bytes())),map_location='cpu',weights_only=False)['policy'];assert S.digest(state)==plan['policy_hash'];p=S.policy(state).eval().requires_grad_(False)
        (OUT/'native').mkdir(exist_ok=True);native,create=S.native_setup(OUT/'native');torch.set_num_threads(1)
        for i,job in enumerate(plan['jobs']):
            env,obs,_=R.task_env(create,job);renderer=mujoco.Renderer(env.model,height=240,width=320);camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
            xy=np.asarray(env._pose()[:2]);goal=env.goal_xy.copy();camera.lookat[:]=[*((xy+goal)/2),.15];camera.distance=job['case']['distance_m']*1.25+1.5;camera.azimuth=90;camera.elevation=-70
            cases.append(dict(index=i,job=job,env=env,render_data=mujoco.MjData(env.model),obs=obs,renderer=renderer,camera=camera,goal=goal,scene=C.model_hash(env),records=[],steps=0,done=False,success=False,collision=False,timeout=False,remaining=float(env._goal_state()[1])))
        movie=OUT/'stable_3m_4m_5m_fresh_mujoco.mp4';assert not movie.exists()
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','5','-i','-','-an','-c:v','libx264','-preset','fast','-b:v','300k','-maxrate','300k','-bufsize','600k','-pix_fmt','yuv420p','-fs',str(2*1024**2),str(movie)]
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=(OUT/'encoder.log').open('wb'))
        def frame():
            nonlocal frame_count
            pics=[]
            for c in cases:
                render=c['renderer'];mujoco.mj_copyData(c['render_data'],c['env'].model,c['env'].data);render.update_scene(c['render_data'],camera=c['camera']);scene=render.scene;g=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.12,.12,.12]),np.array([*c['goal'],.45]),np.eye(3).ravel(),np.array([.05,1.,.2,1.],dtype=np.float32));scene.ngeom+=1
                pic=render.render().copy();cv2.rectangle(pic,(0,0),(320,42),(15,20,28),-1);case=c['job']['case'];label=f"{case['distance_m']:.0f}m {case.get('side','')} task{c['index']}"
                state='GOAL' if c['success'] else 'CONTACT' if c['collision'] else 'TIMEOUT' if c['timeout'] else 'RUNNING'
                cv2.putText(pic,label,(8,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(pic,f"{c['steps']*.002:.2f}s | {c['remaining']:.3f}m | {state}",(8,33),cv2.FONT_HERSHEY_SIMPLEX,.36,(230,240,235),1,cv2.LINE_AA);pics.append(pic)
            grid=np.concatenate([np.concatenate(pics[:4],axis=1),np.concatenate(pics[4:],axis=1)],axis=0);encoder.stdin.write(grid.tobytes());frame_count+=1
            if frame_count==1:cv2.imwrite(str(OUT/'first_frame.jpg'),cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
        frame()
        for tick in range(234):
            guard()
            for c in cases:
                if c['done']:continue
                env=c['env'];cap=c['job']['case']['step_cap'];x=C.observation(c['obs'],1-c['steps']/cap)
                before=c['steps'];d0=float(env._goal_state()[1]);pose=np.asarray(env._pose()[:3])
                with torch.no_grad():action=p(torch.tensor(x[None])).numpy()[0].astype(np.float32)
                for _ in range(min(100,cap-c['steps'])):
                    obs,_,term,trunc,info=native.step_checked(env,action);c['steps']+=1;physics+=1
                    if term or trunc:break
                c.update(obs=obs,done=bool(term or trunc),success=bool(info['reach_goal']),collision=bool(info['collision']),timeout=bool(trunc),remaining=float(env._goal_state()[1]))
                n=c['steps']-before;d1=c['remaining'];discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
                reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
                c['records'].append((x,action,np.zeros(2,dtype=np.float32),0.,n,C.observation(obs,1-c['steps']/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
            frame();C.save(OUT/'status.json',dict(state='FRESH_NATIVE_VIDEO',physics_steps=physics,frames=frame_count,elapsed_s=time.monotonic()-start))
            if all(c['done'] for c in cases):break
        assert all(c['done'] for c in cases);encoder.stdin.close();assert encoder.wait(timeout=30)==0
        centers=[r for r in map(json.loads,(ROOT/'stage/runs/t245_multitask_pcgrad/development.jsonl').read_text().splitlines()) if r['job']['delta_x_m']==r['job']['delta_y_m']==r['job']['delta_yaw_deg']==0.]
        assert len(centers)==len(cases)==8
        for c,expected in zip(cases,centers):
            assert c['job']['case']['id']==expected['job']['case']['id']
            names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'];arrays={k:np.asarray([r[i] for r in c['records']]) for i,k in enumerate(names)};arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8);hashes={k:S.digest(a) for k,a in arrays.items()};assert hashes==expected['array_hashes']
            assert c['steps']==expected['physics_steps'] and R.arrays_hash(R.integration(c['env']))==expected['integration_sha256']
            assert np.array_equal(c['goal'],c['env'].goal_xy) and C.model_hash(c['env'])==c['scene']
            assert c['success'] and not c['collision'];verified.append(dict(task=c['index'],case=c['job']['case']['id'],success=c['success'],collision=c['collision'],physics_steps=c['steps'],remaining_m=c['remaining'],array_hashes=hashes,integration_sha256=expected['integration_sha256'],exact_to_T245=True))
        ffprobe=shutil.which('ffprobe')
        if ffprobe:
            probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(movie)]));assert int(probe['streams'][0]['nb_read_frames'])==frame_count
        assert movie.stat().st_size<2*1024**2;reason='FRESH_VIDEO_VERIFIED';guard()
    except BaseException:reason='FAILED';C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),physics_steps=physics,frames=frame_count))
    finally:
        for c in cases:c['renderer'].close();c['env'].close()
        if encoder is not None and encoder.poll() is None:encoder.kill();encoder.wait()
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    result=dict(state='COMPLETE' if reason=='FRESH_VIDEO_VERIFIED' else 'FAILED',stop_reason=reason,physics_steps=physics,cases=verified,frames=frame_count,navigation_training_updates=0,fresh_neural_rollouts=True,all_native_envs_closed=True,all_encoder_processes_exited=True,visual_goal_marker_only=True,nominal_playback_speed=1.,elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),movie=str(OUT/'stable_3m_4m_5m_fresh_mujoco.mp4'))
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
