"""Preserve canonical wide/radial factory while retaining legacy field resets.

No historical module is edited. Legacy field24 has its own recorded narrower
scene despite wide_course=True; preserve its actual bytes under legacy IDs.
A genuinely widened global-field dataset would need new explicit task IDs.
"""
import copy,math
import numpy as np


def setup(F,folder,p,canonical_setup):
    module,native,create,w,reward=canonical_setup(folder,p)
    canonical_task=F.E.N.K.R.task_env
    def task_env(create,job):
        if 'fullfield_start_pose' not in job:
            # Critical: use the factory installed by setup, which applies the
            # wide obstacle shift and radial goal BEFORE any policy decision.
            return canonical_task(create,job)
        # Match the original T502 actually recorded field geometry; do not
        # silently widen that dataset while fixing old wide6/wide8 construction.
        factory_job=copy.deepcopy(job);factory_job['case']['distance_m']=6.
        env,obs,actual=F.BASE_TASK(create,factory_job)
        import mujoco
        pose=np.asarray(job['fullfield_start_pose'],float);goal=np.asarray(job['fullfield_goal_xy'],float)
        assert np.isfinite(pose).all() and np.isfinite(goal).all() and np.abs(pose[:2]).max()<7 and np.abs(goal).max()<7
        env.data.qpos[:2]=pose[:2];env.data.qpos[3:7]=[math.cos(pose[2]/2),0.,0.,math.sin(pose[2]/2)]
        env.data.qvel[:]=0.;env.data.ctrl[:]=0.;env.data.time=0.;env.step_count=0;env.prev_action=np.zeros(2,np.float32)
        mujoco.mj_forward(env.model,env.data);env.history.clear();obs=np.asarray(env._observation(update_history=False),np.float32)
        before=F.E.N.K.R.integration(env).copy();obs=w.alternate(env,goal,w.get(env))
        assert np.array_equal(before,F.E.N.K.R.integration(env))
        from reset_contract import check_history
        check_history(*w.get(env),pose,goal);assert not env._collision();env.max_steps=job['case']['step_cap']
        return env,obs,actual
    F.E.N.K.R.task_env=task_env
    return module,native,create,w,reward
