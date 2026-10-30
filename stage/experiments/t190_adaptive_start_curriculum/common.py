"""Stage native-command policy and reset-only full-state snapshots."""
from pathlib import Path
from collections import deque
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=REPO/'stage/runs/t190_adaptive_start_curriculum'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

CORE=load(REPO/'stage/current/outputs/stage_t148/td3_core_goal_contrast.py','frontier_policy_core')

class LegacyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.body=CORE.Actor(113,2);self.native_head=torch.nn.Linear(2,2)
        self.logstd=torch.nn.Parameter(torch.tensor([-.69314718,-.69314718]))
    def forward(self,x):return self.native_head(self.body(x))
    def initialize(self,old):
        state={k:v.clone() for k,v in old.items()};state['net.0.weight']=torch.nn.functional.pad(state['net.0.weight'],(0,1))
        self.body.load_state_dict(state)
        with torch.no_grad():
            self.native_head.weight.copy_(torch.diag(torch.tensor([.5,1.])))
            self.native_head.bias.copy_(torch.tensor([.5,0.]))
    def distribution(self,x):return torch.distributions.Normal(self(x),self.logstd.clamp(-1.6,0.).exp())

def observation(obs,remaining):return np.r_[np.asarray(obs,dtype=np.float32),np.float32(np.clip(remaining,0,1))].astype(np.float32)
def action_clip(a,arm):
    a=np.asarray(a,dtype=np.float32).copy();a[0]=np.clip(a[0],0 if arm=='legacy_range' else -1,1);a[1]=np.clip(a[1],-1,1);return a

def native_setup(folder):
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(folder);os.environ['C22_SCREEN_ONLY']='0'
    t=load(REPO/'stage/current/outputs/stage_t148/train.py','frontier_native_source')
    helper=t.load_module(t.HELPER,'frontier_native_helper');legacy=t.load_module(t.STAGE,'frontier_native_legacy')
    for p,h in legacy.EXPECTED.items():assert sha(p)==h,p
    runner=legacy.module(legacy.SRC,'frontier_native_runner');_,v,mod,prep,_=runner.load_native();native=prep[3]
    make=legacy.make_short_task(runner.make_task)
    def create(case):
        env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,case['seed'],case['angle_deg'],case['distance_m'])
        if renderer is not None:renderer.close()
        assert abs(float(env.model.opt.timestep)-.002)<1e-10 and not env._collision()
        return env,np.asarray(obs,dtype=np.float32),actual
    return native,create

def model_hash(env):
    h=hashlib.sha256()
    for name in ('geom_pos','geom_size','geom_contype','geom_conaffinity','body_pos','actuator_gainprm','dof_damping'):
        h.update(np.asarray(getattr(env.model,name)).tobytes())
    return h.hexdigest()

def capture(env,obs):
    import mujoco
    state=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    attrs=copy.deepcopy({k:v for k,v in vars(env).items() if k not in ('model','data')})
    return {'physics':state,'attrs':attrs,'obs':np.asarray(obs,dtype=np.float32).copy(),'model_hash':model_hash(env)}

def restore(env,snapshot):
    import mujoco
    assert model_hash(env)==snapshot['model_hash']
    for k,v in snapshot['attrs'].items():setattr(env,k,copy.deepcopy(v))
    mujoco.mj_setState(env.model,env.data,snapshot['physics'],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(env.model,env.data)
    # Preserve the recorded integration warm-start after rebuilding derived kinematics.
    mujoco.mj_setState(env.model,env.data,snapshot['physics'],mujoco.mjtState.mjSTATE_INTEGRATION)
    return snapshot['obs'].copy()

def write_snapshot(p,s):
    np.savez_compressed(p,physics=s['physics'],attrs=np.frombuffer(pickle.dumps(s['attrs'],protocol=5),dtype=np.uint8),
                        obs=s['obs'],model_hash=np.asarray(s['model_hash']))
def read_snapshot(p):
    with np.load(p,allow_pickle=False) as z:return {'physics':z['physics'].copy(),'attrs':pickle.loads(z['attrs'].tobytes()),'obs':z['obs'].copy(),'model_hash':str(z['model_hash'])}

import math
def perturb_anchor(env,obs,job):
 import mujoco
 dx,dy,yaw_delta=job['delta_x_m'],job['delta_y_m'],job['delta_yaw_deg']
 if dx==dy==yaw_delta==0:return obs
 yaw=float(env._pose()[2])+math.radians(yaw_delta)
 env.data.qpos[:2]+=[dx,dy];env.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
 mujoco.mj_forward(env.model,env.data);env.prev_action=np.zeros(2,dtype=np.float32);env.step_count=0;env.history.clear();env.prev_goal_dist=float(env._goal_state()[1])
 return np.asarray(env._observation(update_history=False),dtype=np.float32)


class Policy(LegacyPolicy):
    def __init__(self):
        super().__init__()
        self.direct_head=torch.nn.Linear(600,2)
        torch.nn.init.zeros_(self.direct_head.weight);torch.nn.init.zeros_(self.direct_head.bias)
    def forward(self,state):
        features=self.body.net[:4](state)
        return self.native_head(self.body.net[4:](features))+self.direct_head(features)
    def load_parent(self,state):
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={'direct_head.weight','direct_head.bias'} and not result.unexpected_keys
        assert torch.count_nonzero(self.direct_head.weight)==0 and torch.count_nonzero(self.direct_head.bias)==0
