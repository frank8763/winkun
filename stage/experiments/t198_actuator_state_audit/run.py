"""Frozen steering-state observability and reset-only matched diagnostic probes."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import time,traceback,mujoco
from common import *

def main():
    assert not (OUT/'status.json').exists(),'No duplicate audit'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    actor=Policy().eval().requires_grad_(False);state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];actor.load_state_dict(state)
    native,create=native_setup(OUT/'native');deadline=time.monotonic()+plan['wall_seconds'];spent=0;episodes=[];env=None
    from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import get_last_step_record
    def joint_ids(env):
        ids=[int(env.model.actuator_trnid[getattr(env,n),0]) for n in ['id_lf_steer','id_rf_steer']]
        return [int(env.model.jnt_qposadr[j]) for j in ids],[int(env.model.jnt_dofadr[j]) for j in ids]
    try:
        save(OUT/'status.json',{'state':'AUDITING','optimizer_updates':0})
        for case in plan['cases']:
            env,obs,actual=create(case);env.max_steps=case['step_cap'];addresses,_=joint_ids(env);steps=0;term=trunc=False;rows=[];decision_errors=[];previous=np.zeros(2)
            while not(term or trunc):
                assert time.monotonic()<deadline
                if steps%100==0:
                    decision_errors.append(np.asarray(env.data.qpos[addresses])-previous)
                    with torch.no_grad():action=action_clip(actor(torch.tensor(observation(obs,1-steps/case['step_cap']).reshape(1,-1))).numpy()[0],'full_range')
                obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;spent+=1
                applied=get_last_step_record(env.data)['last_applied'];target=np.array([applied['lf'],applied['rf']]);actual_angles=np.asarray(env.data.qpos[addresses]).copy()
                rows.append([steps,*env._pose(),*actual_angles,*target,*action]);previous=target
            trace=np.asarray(rows);errors=trace[:,7:9]-trace[:,9:11];decision_errors=np.asarray(decision_errors)
            path=OUT/(case['side']+'_actuator_trace.npz');np.savez_compressed(path,trace=trace.astype(np.float32),columns=np.asarray(['step','x','y','yaw','vx','vy','yaw_rate','lf_angle','rf_angle','lf_target','rf_target','native_throttle','native_steering']))
            episodes.append({'case':case['id'],'physics_steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'tracking_absolute_error_quantiles_rad':np.quantile(np.abs(errors),[.5,.9,.99,1]).tolist(),'decision_boundary_error_quantiles_rad':np.quantile(np.abs(decision_errors),[.5,.9,.99,1]).tolist(),'trace_sha256':sha(path)})
            env.close();env=None
        probes=[]
        for sign in [-1,1]:
            env,_,_=create(plan['cases'][0]);obs=restore(env,read_snapshot(plan['snapshot']));qpos_addr,qvel_addr=joint_ids(env)
            # Validity diagnostic initialization only; no state writes after stepping begins.
            env.data.qpos[qpos_addr]=sign*.2;env.data.qvel[qvel_addr]=0.;mujoco.mj_forward(env.model,env.data)
            obs=np.asarray(env._observation(update_history=False),dtype=np.float32);assert not env._collision();initial_obs=obs.copy();initial_pose=np.asarray(env._pose()).copy()
            x=observation(obs,1-plan['snapshot_elapsed']/plan['cases'][0]['step_cap'])
            with torch.no_grad():command=action_clip(actor(torch.tensor(x.reshape(1,-1))).numpy()[0],'full_range')
            for _ in range(100):
                assert time.monotonic()<deadline
                obs,_,t,u,_=native.step_checked(env,command);spent+=1;assert not(t or u)
            probes.append({'initial_obs':initial_obs,'initial_pose':initial_pose,'command':command,'final_pose':np.asarray(env._pose()).copy(),'final_obs':np.asarray(obs).copy(),'joint_angles':sign*.2});env.close();env=None
        before=float(np.abs(probes[0]['initial_obs']-probes[1]['initial_obs']).max());command_error=float(np.abs(probes[0]['command']-probes[1]['command']).max())
        pose_difference=(probes[0]['final_pose']-probes[1]['final_pose']).tolist();position_difference=float(np.linalg.norm(np.asarray(pose_difference)[:2]))
        columns=[57,58,74,75,86,87,98,99,110,111];weights=state['body.net.0.weight'][:,columns];assert torch.count_nonzero(weights)==0
        assert all(torch.equal(state[k],v) for k,v in actor.state_dict().items())
        assert spent<=plan['max_physics']
        result={'state':'COMPLETE','episodes':episodes,'physics_steps':spent,'optimizer_updates':0,'history_action_columns_all_zero':True,'reset_probe':{'initial_observation_max_difference':before,'initial_command_max_difference':command_error,'same_observation_established':before<1e-6,'joint_angles_rad':[-.2,.2],'same_neural_command_100physics_steps':100,'final_pose_difference':pose_difference,'final_xy_difference_m':position_difference,'final_observation_max_difference':float(np.abs(probes[0]['final_obs']-probes[1]['final_obs']).max()),'scope':'Constructed reset-only steering state intervention, same initial base pose/velocity and cachedhistory. First100physicssteps use same initial neural command only if initialobservations/commands match. This is an observability probe, not navigation deployment, source of successful routes or proof of failure cause.'},'automatic_promotion':False,'incumbent':'T159','scope':plan['scope']}
        save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result,indent=2))
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','physics_steps':spent,'traceback':traceback.format_exc()});raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
