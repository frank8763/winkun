"""One frozen LEFT reproduction, native contact witness and complete-state check."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import time,traceback,mujoco
from common import *

def integration(env):
    a=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,a,mujoco.mjtState.mjSTATE_INTEGRATION);return a

def contacts(env):
    rows=[]
    for con in env.data.contact[:env.data.ncon]:
        a,b=int(con.geom1),int(con.geom2);ra=int(env.model.geom_bodyid[a]) in env.robot_body_ids;rb=int(env.model.geom_bodyid[b]) in env.robot_body_ids
        if ra==rb or float(con.dist)>.002:continue
        other=b if ra else a;robot=a if ra else b
        name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,other) or '';body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[other])) or ''
        if any(x in (name+' '+body).lower() for x in ['floor','ground','plane']):continue
        rows.append({'robot_geom':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,robot),'other_geom':name,'distance_m':float(con.dist)})
    return rows

def main():
    assert not (OUT/'diagnostic.json').exists(),'No repeat diagnostic'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=Policy().eval().requires_grad_(False);state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];policy.load_state_dict(state)
    native,create=native_setup(OUT/'diagnostic_native');case=plan['primary_cases'][6]
    old=next(json.loads(s) for s in Path(plan['parent_evaluation']).read_text().splitlines() if json.loads(s)['group']=='primary' and json.loads(s)['case']==case['id'])
    capture_at=((old['physics_steps']-1000)//100)*100;deadline=time.monotonic()+120
    env=clone=None;steps=spent=0;term=trunc=False;records=[];future_physics=[];future_obs=[];snapshot_path=OUT/'left_frontier_fullstate.npz'
    try:
        env,obs,actual=create(case);env.max_steps=case['step_cap'];goal=env.goal_xy.copy()
        while not(term or trunc):
            assert time.monotonic()<deadline
            if steps==capture_at:
                write_snapshot(snapshot_path,capture(env,obs));saved_obs=obs.copy()
            with torch.no_grad():action=action_clip(policy(torch.tensor(observation(obs,1-steps/case['step_cap']).reshape(1,-1))).numpy()[0],'full_range')
            records.append((steps,observation(obs,1-steps/case['step_cap']),action.copy(),np.asarray(env._pose())))
            for _ in range(min(100,case['step_cap']-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;spent+=1
                if capture_at<steps<=capture_at+200:future_physics.append(integration(env));future_obs.append(obs.copy())
                if term or trunc:break
        assert steps==old['physics_steps'] and info['collision'] and not info['reach_goal'] and np.array_equal(goal,env.goal_xy)
        contact_rows=contacts(env);assert contact_rows
        path=OUT/'left_baseline_trace.npz';np.savez_compressed(path,steps=np.asarray([r[0] for r in records]),states=np.asarray([r[1] for r in records]),actions=np.asarray([r[2] for r in records]),poses=np.asarray([r[3] for r in records]))
        clone,_,_=create(case);obs2=restore(clone,read_snapshot(snapshot_path));assert np.array_equal(obs2,saved_obs);pe=oe=0.
        for i in range(200):
            absolute=capture_at+i
            if i%100==0:
                with torch.no_grad():a=action_clip(policy(torch.tensor(observation(obs2,1-absolute/case['step_cap']).reshape(1,-1))).numpy()[0],'full_range')
            obs2,_,t,u,_=native.step_checked(clone,a);obs2=np.asarray(obs2,dtype=np.float32);spent+=1;assert not(t or u)
            pe=max(pe,float(np.abs(integration(clone)-future_physics[i]).max()));oe=max(oe,float(np.abs(obs2-future_obs[i]).max()))
        assert pe==oe==0
        assert all(torch.equal(state[k],v) for k,v in policy.state_dict().items())
        result={'state':'PASS','baseline_steps':steps,'diagnostic_physics':spent,'remaining_m':float(env._goal_state()[1]),'contacts':contact_rows,'final_pose':list(env._pose()),'final_lidar_min_m':float(obs[:48].min()*6),'snapshot_elapsed_steps':capture_at,'snapshot_path':str(snapshot_path),'snapshot_sha256':sha(snapshot_path),'restore_future_steps':200,'physics_max_error':pe,'observation_max_error':oe,'actor_unchanged':True,'optimizer_updates':0,'scope':'Frozen diagnostic only; snapshot never used as runtime waypoint or primary5m shortcut.'}
        save(OUT/'diagnostic.json',result);print(json.dumps(result,indent=2))
    except BaseException:
        save(OUT/'diagnostic_failure.json',{'state':'FAILED','physics_steps':spent,'traceback':traceback.format_exc()});raise
    finally:
        if env is not None:env.close()
        if clone is not None:clone.close()

if __name__=='__main__':main()
