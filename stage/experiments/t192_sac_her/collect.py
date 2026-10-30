"""SAC sampling from native episodes with witnessed physical history metadata."""
import os,sys,time,traceback,math
from common import *

def run(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        plan=json.loads((HERE/'protocol.json').read_text());native,create=native_setup(folder)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        original=Base._history_frame;events={}
        def hook(env,lidar,distance,angle):
            f=original(env,lidar,distance,angle);events.setdefault(id(env),deque(maxlen=16)).append((f.copy(),np.asarray(env._pose()[:3]),env.goal_xy.copy()));return f
        Base._history_frame=hook
        def history(env):
            records=[]
            for frame in env.history:
                match=next((r for r in reversed(events[id(env)]) if np.array_equal(r[0],frame)),None);assert match is not None;records.append(match)
            return np.asarray([r[1] for r in records]),np.asarray([r[2] for r in records])
        policy=Policy().eval().requires_grad_(False)
        while True:
            command=pipe.recv()
            if command is None:break
            job,weights=command;policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()});rng=np.random.default_rng(job['seed'])
            if env is not None:env.close();env=None
            env,obs,actual=create(job['case']);cap=job['case']['step_cap'];env.max_steps=cap
            if job.get('snapshot_side'):
                obs=restore(env,read_snapshot(plan['anchor_sources'][job['snapshot_side']]['snapshot_path']));env.max_steps=cap;events[id(env)]=deque(maxlen=16)
                for f in env.history:events[id(env)].append((f.copy(),np.asarray(env._pose()[:3]),env.goal_xy.copy()))
            goal=env.goal_xy.copy();model=model_hash(env);steps=0;term=trunc=False;records=[];physical=[];flags=[];nearest=float(env._goal_state()[1])
            while not(term or trunc):
                assert time.monotonic()<deadline
                hp,hg=history(env);pose=np.asarray(env._pose()[:3]);state=observation(obs,1-steps/cap)
                with torch.no_grad():
                    x=torch.tensor(state.reshape(1,-1));mu=policy.latent_mean(x).numpy()[0]
                    z=mu+policy.logstd.clamp(-3.,0.).exp().numpy()*rng.normal(size=2) if job['stochastic'] else mu
                    action=np.tanh(z).astype(np.float32)
                before=steps
                for _ in range(min(100,cap-steps)):
                    t0=float(env.data.time);obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1
                    assert abs(float(env.data.time)-t0-.002)<1e-7
                    physical.append(np.asarray(env._pose()[:2]));flags.append([bool(info['collision']),bool(info['reach_goal']),bool(trunc)])
                    nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
                nhp,nhg=history(env)
                records.append((before,state,action.copy(),steps-before,observation(obs,1-steps/cap),pose,hp,hg,np.asarray(env._pose()[:3]),nhp,nhg))
            assert model_hash(env)==model and np.array_equal(goal,env.goal_xy)
            row={'id':job['id'],'arm':job['arm'],'case':job['case']['id'],'snapshot_side':job.get('snapshot_side'),'physics_steps':steps,'decisions':len(records),'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'nearest_m':nearest,'remaining_m':float(env._goal_state()[1]),'actual_seed':int(actual),'stochastic':job['stochastic']}
            if job.get('collect',False):
                arrays={k:np.asarray([r[i] for r in records]) for i,k in enumerate(['steps','states','actions','native_counts','next_states','poses','history_poses','history_goals','next_poses','next_history_poses','next_history_goals'])}
                arrays['physical_xy']=np.asarray(physical,dtype=np.float64);arrays['physical_flags']=np.asarray(flags,dtype=np.uint8);arrays['goal_xy']=goal
                path=OUT/'episodes'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,**arrays)
                row['path']=str(path);row['sha256']=sha(path)
            pipe.send(row)
    except(EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
