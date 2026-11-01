"""Frozen safe T255, true6m final goal, reset-only translation with fresh history."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,R,np,torch,policy
import json,math,mujoco,time


def initial(create,job):
    env,obs,_=R.task_env(create,job['base_job'])
    try:
        base=L.packet(env,obs,0,28000,np.zeros(2,dtype=np.float32),0)
        with np.load(job['reference_initial_path']) as z:assert np.array_equal(base,z['states'][0])
        original_pose=np.array(env._pose()[:3]);goal=env.goal_xy.copy();before=R.integration(env);qpos=env.data.qpos.copy()
        old_distance=float(np.linalg.norm(goal-original_pose[:2]));assert abs(old_distance-6)<1e-5
        desired=goal-(goal-original_pose[:2])/old_distance*job['remaining_distance_m']
        warm=env.data.qacc_warmstart.copy();env.data.qpos[:2]=desired
        mujoco.mj_forward(env.model,env.data);env.data.qacc_warmstart[:]=warm
        assert np.array_equal(env.data.qpos[2:],qpos[2:]),'Do not silently flatten roll/pitch or change joints'
        after=R.integration(env);diff=np.flatnonzero(before!=after)
        assert len(diff)<=2 and np.array_equal(np.sort(after[diff]),np.sort(env.data.qpos[:2]))
        env.prev_action=np.zeros(2,dtype=np.float32);env.step_count=0;env.history.clear();env.prev_goal_dist=float(env._goal_state()[1])
        obs=np.asarray(env._observation(update_history=False),dtype=np.float32)
        x=L.packet(env,obs,0,28000,np.zeros(2,dtype=np.float32),0)
        actual=float(env._goal_state()[1]);assert abs(actual-job['remaining_distance_m'])<1e-8
        for frame in env.history:
            assert abs(float(frame[8])-actual/10)<1e-6
            assert abs(float(frame[9])-float(env._goal_state()[2])/math.pi)<1e-6
        assert np.array_equal(goal,env.goal_xy) and x.shape==(120,) and x[112]==1 and np.array_equal(x[117:],np.zeros(3,dtype=np.float32))
        data=dict(id=job['id'],side=job['side'],remaining_distance_m=job['remaining_distance_m'],valid=not env._collision(),
            original_zero_translation120_exact=True,only_native_qpos_XY_changed=True,original_pose=original_pose.tolist(),
            reset_pose=list(env._pose()[:3]),goal_xy=goal.tolist(),initial120_sha256=S.digest(x),
            integration_sha256=R.arrays_hash(after),model_sha256=C.model_hash(env),history_fresh_at_new_pose=True,
            reset_initial_state_not_historical_complete6m_episode=True)
        return env,obs,data
    except BaseException:
        env.close();raise


def rollout(env,obs,p,meter,label):
    cap=28000;previous=np.zeros(2,dtype=np.float32);previous_count=0;steps=0;term=trunc=False;info={};records=[];error=None
    scene=C.model_hash(env);goal=env.goal_xy.copy()
    try:
        while not(term or trunc) and steps<cap and meter.running():
            meter.hold();x=L.packet(env,obs,steps,cap,previous,previous_count);pose=np.array(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():action=p.step(torch.tensor(x[None]))[0][0].numpy()
            for _ in range(min(100,cap-steps)):
                if not meter.running():break
                obs,_,term,trunc,info=R_NATIVE.step_checked(env,action);steps+=1;meter.physics+=1
                if term or trunc:break
            n=steps-before
            if n==0:break
            d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=L.packet(env,obs,steps,cap,action,n)
            records.append((x,action,np.zeros(2,dtype=np.float32),0.,n,nxt,pose,np.array(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
            previous=action.copy();previous_count=n
    except BaseException as exc:
        import traceback
        error=traceback.format_exc()
    finally:
        integration=R.arrays_hash(R.integration(env));remaining=float(env._goal_state()[1])
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
        env.close()
    names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
    arrays={k:np.array([r[i] for r in records]) for i,k in enumerate(names)}
    arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
    row=dict(id=label,physics_steps=steps,complete=bool(term or trunc),success=bool(info.get('reach_goal')),
        collision=bool(info.get('collision')),timeout=bool(trunc),administratively_censored=not(term or trunc),
        remaining_m=remaining,array_hashes={k:S.digest(v) for k,v in arrays.items()},integration_sha256=integration,
        error=error,no_Actor_update=True)
    return row,arrays
