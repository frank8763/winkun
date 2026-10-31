"""Frozen Stage acceptance with explicit physical-start validity, no scene edits."""
from pathlib import Path
import sys,os,json,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t229_valid_start_acceptance'
sys.path.insert(0,str(ROOT/'stage/experiments/t223_coverage_repair'))
import repair_support as B
S,C,R,np,torch=B.S,B.C,B.R,B.np,B.torch
used,digest,state_np=S.used,S.digest,S.state_np

def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);pipe.send(dict(ready=True,pid=os.getpid()));p=None;ph=None
        while True:
            msg=pipe.recv()
            if msg is None:break
            kind,j=msg
            if kind=='validate':
                env,obs,_=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j)
                row=dict(id=j['id'],valid_start=not env._collision(),physics_steps=0,scene_sha256=C.model_hash(env),initial_pose=list(map(float,env._pose()[:3])),goal_xy=env.goal_xy.tolist())
                env.close();env=None;pipe.send(row)
            else:
                assert kind=='evaluate' and digest(j['policy_np'])==j['policy_hash']
                if ph!=j['policy_hash']:p=S.load_np(R.Policy(),j['policy_np']).eval().requires_grad_(False);ph=j['policy_hash']
                data=B.generate(j,p,native,create,deadline);data['row'].update(valid_start=True,group=j['group'],policy_hash=ph,job=j)
                data['row']['job']={k:v for k,v in j.items() if k!='policy_np'};pipe.send(data)
    except EOFError:pass
    except BaseException:
        error={'error':traceback.format_exc()};C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
