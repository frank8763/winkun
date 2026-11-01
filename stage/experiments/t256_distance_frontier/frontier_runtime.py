"""Frozen Stage distance-frontier audit; never modifies a policy or physics."""
from pathlib import Path
import sys,os,json,time,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t255_overnight'))
import mem_runtime as M
Actor,C,R,S,np,torch=M.Actor,M.C,M.R,M.S,M.np,M.torch
def worker(pipe,index,out,plan,deadline):
    folder=Path(out)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder)
        parent=M.read(plan['initial_base'])['policy'];p=Actor(parent,False,2026103501);p.load_state_dict(M.read(plan['checkpoint'])['policy']);p.eval().requires_grad_(False);assert S.digest(p.state_dict())==plan['policy_hash'];pipe.send(dict(ready=True,pid=os.getpid()))
        while True:
            message=pipe.recv()
            if message is None:break
            kind,j=message
            if kind in ['make','inspect']:
                env=None
                try:
                    env,obs,actual=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j);x=M.packet(env,obs,0,j['case']['step_cap'],np.zeros(2,dtype=np.float32),0);valid=not env._collision();pose=np.asarray(env._pose()[:3]);goal_features,dist,angle=env._goal_state()
                    assert np.allclose(x[48:57],goal_features,atol=1e-7) and x.shape==(120,)
                    result=dict(id=j['id'],valid=valid,physics_steps=0,requested_case=j['case'],actual_factory_seed=actual,pose=pose.tolist(),goal_xy=env.goal_xy.tolist(),distance_m=float(dist),goal_angle_rad=float(angle),scene_sha256=C.model_hash(env),observation_sha256=S.digest(x),initial_observation=x.tolist())
                    if kind=='make':
                        assert valid and abs(dist-j['case']['distance_m'])<1e-5
                        path=Path(out)/'tasks'/(j['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();C.write_snapshot(path,C.capture(env,obs));result.update(snapshot=str(path),snapshot_sha256=C.sha(path))
                    pipe.send(result)
                finally:
                    if env is not None:env.close()
            else:
                assert kind=='rollout';item=M.generate(j,p,native,create,deadline);item['row']['policy_hash']=plan['policy_hash'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc());C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
    finally:pipe.close();log.close()
