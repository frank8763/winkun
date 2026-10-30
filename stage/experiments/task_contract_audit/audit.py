"""Stage goal/start continuity audit. Reset geometry only, no simulation steps."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'stage/runs/task_contract_audit'

def main():
    OUT.mkdir(exist_ok=False)
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(OUT)
    os.environ['C22_SCREEN_ONLY']='0';torch.set_num_threads(1)
    path=REPO/'stage/current/outputs/stage_t148/train.py'
    spec=importlib.util.spec_from_file_location('task_contract_source',path)
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    helper=t.load_module(t.HELPER,'task_contract_helper');legacy=t.load_module(t.STAGE,'task_contract_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'task_contract_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
    make=legacy.make_short_task(runner.make_task)
    continuity=[];bank=[]
    for seed,angle in [(2029653601,-17.3),(2029654802,18.8)]:
        for distance in [3.,3.5,4.,4.5,5.]:
            env=renderer=None
            try:
                env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,seed,angle,distance)
                continuity.append({'seed_requested':seed,'angle_deg':angle,'distance_m':distance,
                    'actual_seed':int(actual),'start_pose':list(env._pose()),'goal_xy':env.goal_xy.tolist(),
                    'physics_dt':float(env.model.opt.timestep)})
            finally:
                if renderer is not None:renderer.close()
                if env is not None:env.close()
        env=renderer=None
        try:
            env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,seed,angle,3.)
            original=env.data.qpos.copy();velocity=env.data.qvel.copy();control=env.data.ctrl.copy()
            x,y,yaw,*_=env._pose()
            for distance in [3.,3.5,4.,4.5,5.]:
                for offset in [-30.,-15.,0.,15.,30.]:
                    bearing=angle+offset;direction=yaw+np.deg2rad(bearing)
                    goal=np.array([x,y])+distance*np.array([np.cos(direction),np.sin(direction)])
                    try:
                        env.data.qpos[:2]=goal
                        import mujoco
                        mujoco.mj_forward(env.model,env.data)
                        endpoint_collision=bool(env._collision())
                    finally:
                        env.data.qpos[:]=original;env.data.qvel[:]=velocity;env.data.ctrl[:]=control
                        mujoco.mj_forward(env.model,env.data)
                    bank.append({'template_seed':seed,'template_angle_deg':angle,'template_actual_seed':int(actual),
                        'start_pose':[x,y,yaw],'distance_m':distance,'goal_angle_deg':bearing,
                        'goal_xy':goal.tolist(),'goal_pose_collision':endpoint_collision,
                        'whole_path_vehicle_reachability':'not established'})
            assert np.array_equal(env.data.qpos,original) and np.array_equal(env.data.qvel,velocity) and np.array_equal(env.data.ctrl,control)
        finally:
            if renderer is not None:renderer.close()
            if env is not None:env.close()
    report={'experiment':'STAGE_FIXED_START_GOAL_CONTRACT_AUDIT','native_physics_steps':0,'training_updates':0,
        'factory_distance_continuity':continuity,'fixed_start_final_goal_candidates':bank,
        'endpoint_collision_free_candidates':sum(not r['goal_pose_collision'] for r in bank),
        'candidate_count':len(bank),'planner':False,'waypoints':False,
        'scope':'Final training goal candidates only, no route or ordered subgoals. Exact simulation geometry used only for reset feasibility, as in the existing Stage factory. No navigation success or full reachability claimed.'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'candidates':len(bank),'endpoint_free':report['endpoint_collision_free_candidates'],
        'continuity':[(r['seed_requested'],r['distance_m'],r['actual_seed'],r['start_pose'][:2]) for r in continuity]},indent=2))

if __name__=='__main__':main()
