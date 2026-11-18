"""Stage field reset with periodic yaw comparison; no dynamics changes."""
import copy
import math
import numpy as np


def check_history(hp, hg, ht, pose, goal):
    hp,hg,ht,pose,goal=map(np.asarray,(hp,hg,ht,pose,goal))
    assert hp.shape==(4,3) and hg.shape==(4,2) and ht.shape==(4,)
    assert pose.shape==(3,) and goal.shape==(2,)
    assert all(np.isfinite(a).all() for a in (hp,hg,ht,pose,goal))
    xy=float(np.max(np.abs(hp[:,:2]-pose[:2])))
    dyaw=hp[:,2]-pose[2]
    angular=float(np.max(np.abs(np.arctan2(np.sin(dyaw),np.cos(dyaw)))))
    assert xy<1e-10 and angular<1e-10, dict(xy_error=xy,yaw_error=angular)
    assert np.all(hg==goal) and np.max(np.abs(ht))<1e-10


def field_setup(F, folder, p):
    """Same original operations/quaternions/history, corrected assertion only."""
    module,native,create,w,reward=F.BASE_SETUP(folder,F.E.N.science(p))

    def task_env(create,job):
        factory_job=copy.deepcopy(job)
        if 'fullfield_start_pose' in job:
            factory_job['case']['distance_m']=6.
        env,obs,actual=F.BASE_TASK(create,factory_job)
        if 'fullfield_start_pose' not in job:
            return env,obs,actual
        import mujoco
        pose=np.asarray(job['fullfield_start_pose'],float)
        goal=np.asarray(job['fullfield_goal_xy'],float)
        assert np.isfinite(pose).all() and np.isfinite(goal).all()
        assert np.abs(pose[:2]).max()<7 and np.abs(goal).max()<7
        env.data.qpos[:2]=pose[:2]
        env.data.qpos[3:7]=[math.cos(pose[2]/2),0.,0.,math.sin(pose[2]/2)]
        env.data.qvel[:]=0.;env.data.ctrl[:]=0.;env.data.time=0.
        env.step_count=0;env.prev_action=np.zeros(2,np.float32)
        mujoco.mj_forward(env.model,env.data)
        env.history.clear()
        obs=np.asarray(env._observation(update_history=False),np.float32)
        before=F.E.N.K.R.integration(env).copy()
        obs=w.alternate(env,goal,w.get(env))
        assert np.array_equal(before,F.E.N.K.R.integration(env))
        check_history(*w.get(env),pose,goal)
        assert not env._collision()
        env.max_steps=job['case']['step_cap']
        return env,obs,actual

    F.E.N.K.R.task_env=task_env
    return module,native,create,w,reward
