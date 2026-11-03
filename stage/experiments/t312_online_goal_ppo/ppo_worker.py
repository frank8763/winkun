from ppo_learning import *
import contextlib,importlib.util
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
import t280_worker as W

def worker(pipe,index,p,counts):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;witness=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);q=policy(p).eval().requires_grad_(False)
        spec=importlib.util.spec_from_file_location('t312_history',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base);original=R.task_env;lastenv=None
        def factory(create,j):
            nonlocal lastenv
            env,obs,info=original(create,j);lastenv=env
            if 'active_goal_xy' in j:
                before=R.arrays_hash(R.integration(env));obs=witness.alternate(env,j['active_goal_xy'],witness.get(env));assert before==R.arrays_hash(R.integration(env))
            return env,obs,info
        R.task_env=factory;B.OUT=OUT
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            if j.get('geometry_only'):
                env=None
                try:
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):env,obs,_=factory(create,j);obs=C.perturb_anchor(env,obs,j)
                    meta=witness.get(env);x=T.L.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,np.float32),0)
                    assert np.abs(H.rebuild(x,env._pose()[:3],meta[0],env.goal_xy,meta[1])-x).max()<=2e-6
                    pipe.send(dict(id=j['id'],valid=not env._collision(),physics_steps=0,initial120_sha256=S.digest(x),integration_sha256=R.arrays_hash(R.integration(env)),goal_xy=env.goal_xy.tolist()))
                finally:
                    if env is not None:witness.events.pop(id(env),None);env.close()
                continue
            S.load_np(q,j['policy_np']);assert S.digest(q.state_dict())==j['policy_hash']
            if j['role']=='training':item=B.generate(j,q,native,create,witness,j['deadline_epoch'],counts,index,total)
            else:item=W.generate(j,q,native,create,j['deadline_epoch'],counts,index,total)
            total+=item['row']['physics_steps'];item['row'].update(policy_hash=j['policy_hash'],role=j['role'],arm=j['arm'],round=j.get('round'),goal_family=j.get('goal_family'),draw=j.get('draw'),source_reset_id=j['source_reset_id'],job={k:v for k,v in j.items() if k!='policy_np'})
            if lastenv is not None:witness.events.pop(id(lastenv),None)
            pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),completed_worker_physics=total);C.save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if witness is not None:witness.close()
        pipe.close();log.close()
