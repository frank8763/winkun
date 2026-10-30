"""Independent native Stage simulators; no learner, route or privileged policy input."""
from pathlib import Path
import hashlib
import importlib.util
import math
import os
import signal
import sys
import traceback
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'stage/experiments/t169_value_seed'))
from reward_contract import reward,GAMMA

def run(conn,index,job,seed):
    folder=Path(job)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(folder);os.environ['C22_SCREEN_ONLY']='0'
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('vector_worker_source',REPO/'stage/current/outputs/stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    helper=t.load_module(t.HELPER,'vector_helper');legacy=t.load_module(t.STAGE,'vector_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h,p
    runner=legacy.module(legacy.SRC,'vector_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
    make=legacy.make_short_task(runner.make_task);rng=np.random.default_rng(seed+index*101)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
    env=renderer=None;resets=0;episode_frames=0;task=None
    try:
        while True:
            command=conn.recv()
            if command[0]=='close':break
            if command[0]=='reset':
                if env is not None:env.close();env=None
                phase=int(command[1]);resets+=1
                requested,angle=(2029653601,-17.3) if index%2==0 else (2029654802,18.8)
                env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,requested,angle,5.)
                if renderer is not None:renderer.close();renderer=None
                final_goal=env.goal_xy.copy();original=env.data.qpos.copy()
                canonical=(phase>=2 and resets%4==0)
                import mujoco
                if not canonical:
                    accepted=False
                    lo,hi=[(.5,1.),(1.,2.),(2.,3.5),(3.5,5.)][phase]
                    heading_span=[10.,20.,30.,45.][phase]
                    for attempt in range(64):
                        radius=float(rng.uniform(lo,hi));theta=float(rng.uniform(-math.pi,math.pi))
                        xy=final_goal+radius*np.array([math.cos(theta),math.sin(theta)])
                        yaw=math.atan2(final_goal[1]-xy[1],final_goal[0]-xy[0])+math.radians(float(rng.uniform(-heading_span,heading_span)))
                        env.data.qpos[:]=original;env.data.qpos[:2]=xy;env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
                        env.data.qvel[:]=0.;env.data.ctrl[:]=0.;mujoco.mj_forward(env.model,env.data)
                        if not env._collision() and float(env._lidar48().min()*6)>=.1:
                            accepted=True;break
                    assert accepted,'No collision-free radial start in64 attempts'
                assert np.array_equal(env.goal_xy,final_goal)
                env.prev_action=np.zeros(2,dtype=np.float32);env.step_count=0;env.prev_goal_dist=float(env._goal_state()[1]);env.history.clear()
                obs=env._observation(update_history=False)
                distance=env.prev_goal_dist;env.max_steps=max(14000,int(math.ceil(2*distance/3*14000)))
                episode_frames=0
                assert not env._collision() and abs(float(env.model.opt.timestep)-.002)<1e-10
                task={'seed_requested':requested,'seed_actual':int(actual),'distance_m':distance,'angle_deg':math.degrees(env._goal_state()[2]),
                    'goal_xy':env.goal_xy.tolist(),'start_pose':list(env._pose()[:3]),'phase':phase,'canonical5m':canonical,
                    'reset_only_pose_initialization':True,'final_goal_never_changed_during_episode':True}
                task['signature']=hashlib.sha256(np.round(np.r_[env.data.qpos,env.goal_xy],8).tobytes()).hexdigest()
                conn.send({'obs':np.asarray(obs,dtype=np.float32),'task':task})
            elif command[0]=='step':
                raw=np.asarray(command[1],dtype=np.float32);limit=int(command[2]);assert 1<=limit<=25
                assert raw.shape==(2,) and np.isfinite(raw).all() and abs(raw).max()<=1.00001
                action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                total=0.;terminal=trunc=False
                for n in range(1,limit+1):
                    previous=float(env._goal_state()[1]);time0=float(env.data.time)
                    obs,_,terminal,trunc,info=native.step_checked(env,action)
                    assert abs(float(env.data.time)-time0-.002)<1e-7
                    total+=GAMMA**(n-1)*reward(previous,float(env._goal_state()[1]),info['reach_goal'],info['collision'],terminal)
                    episode_frames+=1
                    if terminal or trunc:break
                episode=None
                if terminal or trunc:
                    episode={'task':task,'physics_steps':episode_frames,'success':bool(info['reach_goal']),
                        'collision':bool(info['collision']),'remaining_m':float(env._goal_state()[1]),
                        'end':'GOAL' if info['reach_goal'] else 'COLLISION' if info['collision'] else 'TRUNCATED' if trunc else 'TERMINATED'}
                conn.send({'obs':np.asarray(obs,dtype=np.float32),'reward':total,'discount':GAMMA**n,
                    'done':float(terminal),'n':n,'episode':episode})
            else:raise RuntimeError('Unknown worker command')
    except (EOFError,SystemExit):pass
    except BaseException:
        try:conn.send({'error':traceback.format_exc()})
        except (OSError,EOFError):pass
        raise
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        conn.close();log.close()
