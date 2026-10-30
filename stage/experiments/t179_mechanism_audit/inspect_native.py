from pathlib import Path
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import sys,inspect,json,hashlib
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[3];OUT=REPO/'stage/runs/t179_mechanism_audit'
sys.path.insert(0,str(REPO/'stage/experiments/t177_frontier_onpolicy'))
import common as C
assert not (OUT/'native_inspection.json').exists()
native,create=C.native_setup(OUT)
case=json.loads((REPO/'stage/experiments/t177_frontier_onpolicy/protocol.json').read_text())['primary_cases'][-2]
env,obs,actual=create(case)
import mujoco
try:
 gid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'real_circle_obs_9')
 command=env._policy_to_commands
 data={'state':'PASS','physics_steps':0,'native_command_owner':command.__func__.__qualname__,
 'native_command_source':inspect.getsourcefile(command.__func__),'native_command_source_sha256':C.sha(inspect.getsourcefile(command.__func__)),
 'native_command_source_text':inspect.getsource(command.__func__),
 'max_steer':env.max_steer,'max_wheel':env.max_wheel,'speed_gain':env.speed_gain,'wheelbase_logging_only':env.wheelbase,
 'obstacle':{'id':int(gid),'type':int(env.model.geom_type[gid]),'cylinder_type':int(mujoco.mjtGeom.mjGEOM_CYLINDER),'size':env.model.geom_size[gid].tolist(),'local_pos':env.model.geom_pos[gid].tolist(),'world_pos':env.data.geom_xpos[gid].tolist(),'body':int(env.model.geom_bodyid[gid]),'rbound':float(env.model.geom_rbound[gid]),'aabb':env.model.geom_aabb[gid].tolist()},
 'original_model_hash':C.model_hash(env),'actual_seed':int(actual),'goal':env.goal_xy.tolist()}
 C.save(OUT/'native_inspection.json',data);print(json.dumps(data,indent=2))
finally:env.close()
