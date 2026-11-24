"""New geometry only at reset; canonical car, sensing and control unchanged."""
import copy,hashlib,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t698_cross_representation_success'))
import t698_native as N
F,Base=N.F,N.C.C
np=F.np

def setup(F,Base,folder,p):
    module,native,create,w,reward=Base.consistent_reset.setup(F,folder,p,Base.CANONICAL_SETUP)
    original=F.E.N.K.R.task_env
    def task_env(create,job):
        env,obs,actual=original(create,job)
        import mujoco
        ids=[i for i in range(env.model.ngeom) if (mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('real_circle_obs_')]
        assert ids,'No circle geoms; do not silently substitute geometry'
        before=F.E.N.K.C.model_hash(env);before_integration=F.E.N.K.R.integration(env).copy()
        rule=p['layout_rules'][job['layout_id']];rng=np.random.default_rng(rule['seed'])
        shifts=rng.uniform(-rule['translation_bound_m'],rule['translation_bound_m'],(len(ids),2))
        for i,delta in zip(ids,shifts):
            assert env.model.geom_bodyid[i]==0,'Movable obstacle body requires different contract'
            env.model.geom_pos[i,:2]+=delta
        mujoco.mj_setConst(env.model,env.data);mujoco.mj_forward(env.model,env.data)
        assert np.array_equal(before_integration,F.E.N.K.R.integration(env))
        assert F.E.N.K.C.model_hash(env)!=before
        # Clear pre-change sensor history; current observation and all history
        # slots must be generated from the new scene before any policy call.
        env.history.clear();obs=np.asarray(env._observation(update_history=False),np.float32)
        obs=w.alternate(env,np.asarray(job['fullfield_goal_xy']),w.get(env))
        hp,hg,ht=w.get(env);pose=np.asarray(env._pose()[:3])
        assert np.max(abs(hp-pose))<1e-10 and np.all(hg==job['fullfield_goal_xy']) and not ht.any()
        assert env.data.time==0 and not env._collision(),'Invalid reset collision'
        assert np.linalg.norm(np.asarray(job['fullfield_goal_xy'])-pose[:2])>.2
        # Endpoint clearance is geometric qualification only; no path solver,
        # oracle route, reachability claim or success-based filtering.
        goal=np.asarray(job['fullfield_goal_xy'])
        for i in ids:assert np.linalg.norm(env.model.geom_pos[i,:2]-goal)>float(env.model.geom_size[i,0])+.35,'Goal endpoint lacks fixed clearance'
        return env,obs,actual
    F.E.N.K.R.task_env=task_env
    return module,native,create,w,reward

class Session(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        assert not self.hidden.any() and self.steps==0 and not self.rows
        if 'initial_contract' in job:
            c=job['initial_contract'];assert self.scene==c['scene_sha256']
            assert F.E.N.K.R.arrays_hash(self.initial)==c['integration_sha256']
            assert F.E.N.K.R.arrays_hash(self.packet())==c['initial120_sha256']
            assert np.array_equal(self.goal,np.asarray(c['goal_xy']))
