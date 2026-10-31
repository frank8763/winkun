"""Frozen deterministic Stage evaluation, compact hashed evidence."""
from pathlib import Path
import os,sys,json,time,traceback,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t222_stability'
sys.path.insert(0,str(ROOT/'stage/experiments/t221_left_success'))
import left_support as L
S,C,R,np,torch=L.S,L.C,L.R,L.np,L.torch
used,digest=L.used,L.digest

def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None;p=None;policy_hash=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);pipe.send({'ready':True,'pid':os.getpid()})
        while True:
            job=pipe.recv()
            if job is None:break
            assert digest(job['policy_np'])==job['policy_hash']
            if policy_hash!=job['policy_hash']:p=S.load_np(R.Policy(),job['policy_np']).eval().requires_grad_(False);policy_hash=job['policy_hash']
            env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job)
            if env._collision():
                pipe.send(dict(id=job['id'],group=job['group'],valid_start=False,physics_steps=0,complete=False,success=False,collision=None,timeout=None));env.close();env=None;continue
            model=C.model_hash(env);goal=env.goal_xy.copy();nearest=initial_d=float(env._goal_state()[1]);initial_pose=list(map(float,env._pose()[:3]));steps=0;cap=job['case']['step_cap'];term=trunc=False;info={};states=[];actions=[];next_states=[]
            while not(term or trunc) and steps<cap and time.time()<deadline:
                x=C.observation(obs,1-steps/cap)
                with torch.no_grad():action=p(torch.tensor(x[None])).numpy()[0].astype(np.float32)
                assert np.isfinite(action).all() and (np.abs(action)<=1).all()
                states.append(x);actions.append(action)
                for _ in range(min(100,cap-steps)):
                    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
                next_states.append(C.observation(obs,1-steps/cap))
            row={k:job[k] for k in ('id','group','case','exact_snapshot','delta_x_m','delta_y_m','delta_yaw_deg','policy_hash')}
            row.update(valid_start=True,complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,
                simulation_s=steps*.002,initial_distance_m=initial_d,initial_pose=initial_pose,remaining_m=float(env._goal_state()[1]),nearest_m=nearest,
                states_sha256=digest(np.asarray(states)),actions_sha256=digest(np.asarray(actions)),next_states_sha256=digest(np.asarray(next_states)),integration_sha256=R.arrays_hash(R.integration(env)),scene_sha256=model,goal_xy=goal.tolist())
            assert C.model_hash(env)==model and np.array_equal(goal,env.goal_xy);env.close();env=None;pipe.send(row)
    except EOFError:pass
    except BaseException:
        error={'error':traceback.format_exc()};C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
