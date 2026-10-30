"""Native behavior episodes with optional fixed neural steering-head perturbation."""
import os,sys,time,traceback,hashlib,math
from common import *

def run(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
    try:
        import mujoco
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);plan=json.loads((HERE/'protocol.json').read_text());native,create=native_setup(folder)
        policy=Policy().eval().requires_grad_(False)
        while True:
            command=pipe.recv()
            if command is None:break
            job,weights=command;policy.placement=job['placement'];policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()});rng=np.random.default_rng(job['seed'])
            expected={k:torch.from_numpy(v.copy()) for k,v in weights.items()}
            assert all(torch.equal(expected[k],v) for k,v in policy.state_dict().items())
            if env is not None:env.close();env=None
            env,obs,actual=create(job['case']);cap=job['case']['step_cap'];env.max_steps=cap
            if job.get('snapshot_side'):
                obs=restore(env,read_snapshot(plan['anchor_sources'][job['snapshot_side']]['snapshot_path']));env.max_steps=cap
            if job.get('perturbation') and any(job['perturbation']):
                dx,dy,dyaw=job['perturbation'];yaw=env._pose()[2]+math.radians(dyaw);env.data.qpos[:2]+=[dx,dy];env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)];mujoco.mj_forward(env.model,env.data)
                env.prev_goal_dist=float(env._goal_state()[1]);obs=np.asarray(env._observation(update_history=False),dtype=np.float32)
            assert not env._collision()
            goal=env.goal_xy.copy();model=model_hash(env);steps=0;term=trunc=False;records=[];nearest=float(env._goal_state()[1])
            while not(term or trunc):
                assert time.monotonic()<deadline
                pose=np.asarray(env._pose()[:3]);state=observation(obs,1-steps/cap)
                with torch.no_grad():
                    x=torch.tensor(state.reshape(1,-1));mu=policy.latent_mean(x).numpy()[0]
                    z=mu+policy.logstd.clamp(-3.,0.).exp().numpy()*rng.normal(size=2) if job['stochastic'] else mu
                    action=np.tanh(z).astype(np.float32)
                before=steps
                for _ in range(min(100,cap-steps)):
                    t0=float(env.data.time);obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;assert abs(float(env.data.time)-t0-.002)<1e-7
                    nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
                records.append((before,state,action.copy(),steps-before,observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[bool(info['collision']),bool(info['reach_goal']),bool(trunc)]))
            assert model_hash(env)==model and np.array_equal(goal,env.goal_xy)
            assert all(torch.equal(expected[k],v) for k,v in policy.state_dict().items())
            row={'id':job['id'],'arm':job['arm'],'case':job['case']['id'],'snapshot_side':job.get('snapshot_side'),'group':job.get('group','near_goal' if job.get('snapshot_side') else 'original5m'),'physics_steps':steps,'decisions':len(records),'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'nearest_m':nearest,'remaining_m':float(env._goal_state()[1]),'actual_seed':int(actual),'stochastic':job['stochastic'],'placement':policy.placement,'behavior_parameters_unchanged_within_episode':True}
            if job.get('collect',False):
                arrays={k:np.asarray([r[i] for r in records]) for i,k in enumerate(['steps','states','actions','native_counts','next_states','poses','next_poses','end_flags'])};arrays['goal_xy']=goal
                arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
                path=OUT/'episodes'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,**arrays);row['path']=str(path);row['sha256']=sha(path)
            pipe.send(row)
    except(EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
