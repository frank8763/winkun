"""Same native Actor execution; auxiliary goals only in separate audit episodes."""
from t298_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t280_worker import generate,LimitedLog
import contextlib,importlib.util,os,traceback

def worker(pipe,index,out,plan,counts):
    folder=Path(out)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;witness=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);p=policy(plan).eval().requires_grad_(False);ph=S.digest(p.state_dict())
        sys.path.insert(0,str(ROOT/'stage/experiments/t295_hindsight_support'))
        spec=importlib.util.spec_from_file_location('t298_readonly_history_witness',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base);original_factory=R.task_env
        def task_env(factory,job):
            env,obs,actual=original_factory(factory,job)
            if 'auxiliary_goal_xy' in job:
                initial=witness.get(env);obs=witness.alternate(env,job['auxiliary_goal_xy'],initial)
            return env,obs,actual
        R.task_env=task_env
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            assert S.digest(j['policy_np'])==j['policy_hash']
            if ph!=j['policy_hash']:S.load_np(p,j['policy_np']);ph=j['policy_hash']
            item=generate(j,p,native,create,j['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps'];item['row']['policy_hash']=ph;pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),completed_worker_physics=total);C.save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if witness is not None:witness.close()
        C.save(folder/'log_accounting.json',dict(bytes_observed=log.total,bytes_written=log.written,limit=65536));pipe.close();log.close()
