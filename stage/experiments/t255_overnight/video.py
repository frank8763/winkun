"""Three fresh native closed-loop videos; require full 5m gate first."""
from mem_runtime import *
import subprocess
def record(folder,jobs,p,expected,deadline,guard):
    import mujoco,cv2,imageio_ffmpeg
    folder=Path(folder);folder.mkdir(exist_ok=True);native,create=S.native_setup(folder);summaries=[]
    for j in jobs:
        guard(1,j['case']['step_cap'],3*1024**2);renderer=None;encoder=None;frames=0;cam=None;render_data=None;movie=folder/(j['source_reset_id']+'.mp4')
        def hook(env,steps,info):
            nonlocal renderer,encoder,frames,cam,render_data
            if renderer is None:
                renderer=mujoco.Renderer(env.model,height=240,width=320);render_data=mujoco.MjData(env.model);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);xy=np.asarray(env._pose()[:2]);cam.lookat[:]=[*((xy+env.goal_xy)/2),.15];cam.distance=8.;cam.azimuth=90;cam.elevation=-70
                encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','5','-i','-','-an','-c:v','libx264','-preset','fast','-b:v','100k','-maxrate','100k','-bufsize','200k','-pix_fmt','yuv420p','-fs',str(1024**2),str(movie)],stdin=subprocess.PIPE,stderr=(folder/(j['id']+'.encoder.log')).open('wb'))
            mujoco.mj_copyData(render_data,env.model,env.data);renderer.update_scene(render_data,camera=cam);scene=renderer.scene;g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.12,.12,.12]),np.array([*env.goal_xy,.45]),np.eye(3).ravel(),np.array([.05,1.,.2,1.],dtype=np.float32));scene.ngeom+=1
            pic=renderer.render().copy();cv2.rectangle(pic,(0,0),(320,40),(15,20,28),-1);label='GOAL' if info['reach_goal'] else 'CONTACT' if info['collision'] else 'RUNNING';cv2.putText(pic,j['source_reset_id'],(6,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);cv2.putText(pic,f'{steps*.002:.2f}s | {label}',(6,32),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);encoder.stdin.write(pic.tobytes());frames+=1
        try:
            item=generate(j,p,native,create,deadline,hook);encoder.stdin.close();assert encoder.wait(timeout=30)==0;r=item['row'];old=expected[j['source_reset_id']]
            assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['success'] and not r['collision'];assert movie.stat().st_size<1024**2
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(movie)]));assert int(probe['streams'][0]['nb_read_frames'])==frames
            summaries.append(dict(row=r,movie=str(movie),sha256=C.sha(movie),frames=frames,fresh_native_arrays_and_integration_exact=True))
        finally:
            if renderer is not None:renderer.close()
            if encoder is not None and encoder.poll() is None:encoder.kill();encoder.wait()
    return summaries
