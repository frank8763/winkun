"""Replay original delivered commands; save only new float64 native XY/flags."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,R,np,torch

NATIVE_ARRAYS=['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards']


def replay(ref,native,create,deadline,guard,on_step):
    import time
    env=None
    with np.load(ref['path']) as z:source={k:z[k].copy() for k in z.files}
    try:
        env,obs,_=R.task_env(create,ref['job']);obs=C.perturb_anchor(env,obs,ref['job']);assert not env._collision()
        goal=env.goal_xy.copy();scene=C.model_hash(env);steps=0;cap=ref['job']['case']['step_cap']
        previous=np.zeros(2,dtype=np.float32);previous_count=0;term=trunc=False;xy=[];flags=[];records=[]
        for k,action in enumerate(source['actions']):
            guard();assert time.time()<deadline and not(term or trunc)
            x=L.packet(env,obs,steps,cap,previous,previous_count);assert np.array_equal(x,source['states'][k])
            pose=np.asarray(env._pose()[:3]);assert np.array_equal(pose,source['poses'][k]);before=steps;d0=float(env._goal_state()[1])
            # Execute the original physical hold, checking any earlier native
            # termination instead of forcing a recorded duration artificially.
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;on_step()
                xy.append(np.asarray(env._pose()[:2],dtype=np.float64));flags.append(int(bool(info['collision']))+2*int(bool(info['reach_goal']))+4*int(bool(trunc)))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=L.packet(env,obs,steps,cap,action,n)
            values=(x,action,n,nxt,pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward)
            records.append(values);previous=action.copy();previous_count=n
        arrays={name:np.asarray([r[i] for r in records]) for i,name in enumerate(NATIVE_ARRAYS)}
        arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        assert all(np.array_equal(v,source[k]) for k,v in arrays.items())
        assert all(S.digest(v)==ref['array_hashes'][k] for k,v in arrays.items())
        assert steps==ref['physics_steps'] and bool(term or trunc)==ref['complete']
        integration=R.arrays_hash(R.integration(env));assert integration==ref['integration_sha256']
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
        physical_xy=np.asarray(xy,dtype=np.float64);physical_flags=np.asarray(flags,dtype=np.uint8)
        ends=source['native_counts'].astype(np.int64).cumsum()
        assert np.array_equal(physical_xy[ends-1],source['next_poses'][:,:2]) and np.all(physical_flags[:-1]==0)
        return dict(physical_xy=physical_xy,physical_flags=physical_flags),source,dict(
            id=ref['id'],physics_steps=steps,complete=ref['complete'],success=ref['success'],collision=ref['collision'],timeout=ref['timeout'],
            native_array_hashes={k:S.digest(v) for k,v in arrays.items()},integration_sha256=integration,
            physical_xy_array_sha256=S.digest(physical_xy),physical_flags_array_sha256=S.digest(physical_flags),
            original_source_sha256=ref['sha256'],scene_sha256=scene,goal_xy=goal.tolist(),
            old_actions_delivered_unchanged=True,not_new_independent_navigation_or_training=True)
    finally:
        if env is not None:env.close()
