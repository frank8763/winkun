"""Render fresh frozen neural-policy runs; the green goal is visual-only."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent

def main():
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT)
    os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('t157_video_env',ROOT.parent/'stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    selection=json.loads((ROOT/'selection.json').read_text())
    assert t.sha(selection['checkpoint'])==selection['sha256']
    core=t.load_module(t.CORE,'t157_video_core');agent=t.create_agent(core)
    ck=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(ck['actor']);agent.actor.eval();agent.actor.requires_grad_(False)
    helper=t.load_module(t.HELPER,'t157_video_helper');legacy=t.load_module(t.STAGE,'t157_video_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'t157_video_runner')
    _,v,mod,prep,_=runner.load_native();N=prep[3];make=legacy.make_short_task(runner.make_task)
    cases=[];encoder=None
    try:
        for seed,angle,label in [(2029653601,-17.3,'LEFT -17.3 deg'),(2029654802,18.8,'HARD RIGHT +18.8 deg')]:
            env,old_renderer,obs,actual=helper.make_goal_task(make,v,mod,N,seed,angle,3.0)
            if old_renderer is not None:old_renderer.close()
            env.max_steps=14000
            render=mujoco.Renderer(env.model,height=480,width=640)
            camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
            start=np.asarray(env._pose()[:2]);goal=np.asarray(env.goal_xy)
            camera.lookat[:]=[(start[0]+goal[0])/2,(start[1]+goal[1])/2,.15]
            camera.distance=4.9;camera.azimuth=90;camera.elevation=-70
            cases.append({'env':env,'renderer':render,'camera':camera,'obs':obs,'label':label,
                'seed':seed,'actual_seed':int(actual),'angle':angle,'steps':0,'done':False,
                'success':False,'collision':False,'remaining':3.0,'goal':goal})
        movie=ROOT/'3m_left_right_verified.mp4'
        encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
            '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10',
            '-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',str(movie)],stdin=subprocess.PIPE)
        def frame():
            pictures=[]
            for c in cases:
                r=c['renderer'];r.update_scene(c['env'].data,camera=c['camera'])
                scene=r.scene;g=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([.20,.025,0.]),np.array([*c['goal'],.035]),
                    np.eye(3).ravel(),np.array([.1,.95,.35,.55],dtype=np.float32))
                scene.ngeom+=1
                # Raise a visual beacon so the goal stays visible above floor/robot geometry.
                g=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([.12,.12,.12]),np.array([*c['goal'],.65]),
                    np.eye(3).ravel(),np.array([.05,1.,.2,1.],dtype=np.float32))
                scene.ngeom+=1
                pic=r.render().copy()
                cv2.rectangle(pic,(0,0),(640,83),(15,20,28),-1)
                cv2.putText(pic,c['label']+' | final goal: 3.0 m',(14,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
                state='GOAL / NO COLLISION' if c['success'] else 'COLLISION' if c['collision'] else 'RUNNING'
                cv2.putText(pic,f"t={c['steps']*.002:.3f}s  remaining={c['remaining']:.3f}m  {state}",(14,49),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,240,235),1,cv2.LINE_AA)
                cv2.putText(pic,'Green beacon = goal; radius 0.20m; frozen neural policy',(14,72),cv2.FONT_HERSHEY_SIMPLEX,.43,(150,245,180),1,cv2.LINE_AA)
                pictures.append(pic)
            encoder.stdin.write(np.concatenate(pictures,axis=1).tobytes())
        frame()
        for global_step in range(1,14001):
            for c in cases:
                if c['done']:continue
                raw=agent.act(c['obs']);action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                obs,_,term,trunc,info=N.step_checked(c['env'],action)
                c.update({'obs':obs,'steps':global_step,'done':bool(term or trunc),
                    'success':bool(info['reach_goal']),'collision':bool(info['collision']),
                    'remaining':float(c['env']._goal_state()[1])})
            if global_step%50==0 or all(c['done'] for c in cases):frame()
            if all(c['done'] for c in cases):break
        encoder.stdin.close();rc=encoder.wait();assert rc==0
        result={'checkpoint_sha256':selection['sha256'],'fresh_neural_rollouts':True,
            'visual_goal_marker_only':True,'training':False,'cases':[
                {k:c[k] for k in ['label','seed','actual_seed','steps','success','collision','remaining']} for c in cases]}
        assert all(c['success'] and not c['collision'] for c in cases)
        (ROOT/'video_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('VIDEO_VERIFIED',json.dumps(result),flush=True)
    finally:
        for c in cases:c['renderer'].close();c['env'].close()
        if encoder is not None and encoder.poll() is None:encoder.terminate()

if __name__=='__main__':main()
