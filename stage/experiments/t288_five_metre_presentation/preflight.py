from run import *
import builtins,dis,types


def main():
    p=json.loads((HERE/'protocol.json').read_text());assert not any((OUT/n).exists() for n in ['status.json','preflight.json','summary.json'])
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    import run as core
    missing=set()
    def walk(code):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(core) and ins.argval not in vars(builtins):missing.add(ins.argval)
        for child in code.co_consts:
            if isinstance(child,types.CodeType):walk(child)
    walk(core.main.__code__);assert not missing,missing
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    used=S.used(ROOT/'stage/runs');assert used-S.used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    torch.set_num_threads(1);actor=M.Actor(M.read(p['initial_base'])['policy'],False,2026103501);actor.load_state_dict(M.read(p['checkpoint'])['policy']);assert S.digest(actor.state_dict())==p['policy_hash']
    native=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody><light pos="0 0 3"/><geom type="plane" size="4 4 .1" rgba=".8 .8 .8 1"/><body pos="0 0 .2"><freejoint/><geom type="box" size=".25 .15 .12" rgba=".1 .4 .8 1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(native);mujoco.mj_forward(native,data);initial=R.arrays_hash(np.asarray(data.qpos),np.asarray(data.qvel));model=copy.copy(native);model.vis.global_.offwidth=1280;model.vis.global_.offheight=840
    assert native.vis.global_.offwidth==640 and native.vis.global_.offheight==480
    copied=mujoco.MjData(model);mujoco.mj_copyData(copied,model,data);renderers=[]
    try:
        renderers=[mujoco.Renderer(model,height=840,width=1280),mujoco.Renderer(model,height=840,width=640)]
        cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[0,0,.1];cam.distance=3.;cam.elevation=-60
        pics=[];start=time.time()
        for renderer in renderers:renderer.update_scene(copied,camera=cam);pics.append(renderer.render().copy())
        canvas=np.zeros((1080,1920,3),dtype=np.uint8);canvas[:]=[13,22,35];canvas[120:960,:1280]=pics[0];canvas[120:960,1280:]=pics[1];cv2.putText(canvas,'T288 SYNTHETIC CODEC TEST',(40,70),cv2.FONT_HERSHEY_SIMPLEX,1.2,(250,250,250),2,cv2.LINE_AA)
        movie=OUT/'preflight_codec.mp4';assert not movie.exists();process=encoder(movie,OUT/'preflight_encoder.log',max_bytes=256*1024)
        for _ in range(3):process.stdin.write(canvas.tobytes())
        process.stdin.close();assert process.wait(timeout=30)==0;assert movie.stat().st_size<256*1024;qc=verify_movie(movie,3)
        assert R.arrays_hash(np.asarray(data.qpos),np.asarray(data.qvel))==initial
        C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,actual_task_environments_created=0,synthetic_model_only=True,
            model_copy_visual_resize_does_not_change_original=True,render_copy_no_physics_mutation=True,ffprobe_not_required=True,
            dual_view_1080p_H264_decode=qc,two_view_render_seconds=time.time()-start,all_nested_run_globals_resolve=True,
            global_if_full=used-S.used(OUT)+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json')))
    finally:
        for renderer in renderers:renderer.close()
    print('T288 PREFLIGHT PASS: synthetic render/copy/1080p H264 encode-decode; zero physical steps and optimizer')


if __name__=='__main__':main()
